# Databricks notebook source
# MAGIC %md
# MAGIC ## Filters on a Fine Tuned Model
# MAGIC This is a RAG chain designed to receive a prompt and optional filters, run it through a model fine tuned for this `text-to-sql` task, and output a sql result. The idea is to perform like **Genie Spaces**, where a list of tables are provided and the endpoint works within those constraints instead of searching throughout the whole lakehouse. This is ideal if there are multiple similar table structures where potential solutions could come from. 
# MAGIC
# MAGIC The prompt comes in with the following structure:\
# MAGIC `{"prompt":"who ordered the most items?", "schema_filter":["mfg_sales_demo.northwind"]}`\
# MAGIC or \
# MAGIC `{"prompt":"who ordered the most items?", "table_filter":["mfg_sales_demo.northwind.orders", "mfg_sales_demo.northwind.order_details"]}`
# MAGIC
# MAGIC It's important to note that filters are optional, so you can submit a prompt like the following and have it consider the whole lakehouse.\
# MAGIC `{"prompt":"who ordered the most items?"}`
# MAGIC
# MAGIC The steps are as follows:
# MAGIC - Use the incoming prompt to do a similarity search on the Query History Vector Search Index.
# MAGIC - Take the list of tables from similar queries and goes against the `t2s_table_definition` endpoint to get table definitions.
# MAGIC - Pass all of these inputs to the prompt function `t2s_build_prompt` to get a fully formatted prompt.
# MAGIC - Get inference from a model (DBRX) endpoint.

# COMMAND ----------

# DBTITLE 1,Install Libraries
# MAGIC %pip install langchain_community langchain_openai
# MAGIC %pip install databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Define Environment Variables
import os
workspaceUrl = spark.conf.get('spark.databricks.workspaceUrl')
workspaceUrl = f"https://{workspaceUrl}"
os.environ['DATABRICKS_HOST'] = workspaceUrl
os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get("robert_mosley", "databricks_token")

# COMMAND ----------

# DBTITLE 1,Define Notebook Variables
catalog = "robert_mosley"
vector_search_endpoint = "one-env-shared-endpoint-7"

# COMMAND ----------

# DBTITLE 1,Define and Test Retriever
from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_community.embeddings import DatabricksEmbeddings

def get_retriever(persist_dir: str = None):
    #Get the vector search index
    vsc = VectorSearchClient(workspace_url=os.environ["DATABRICKS_HOST"], personal_access_token=os.environ["DATABRICKS_TOKEN"])
    vs_index = vsc.get_index(
        endpoint_name=vector_search_endpoint,
        index_name=f"{catalog}.text_to_sql.query_history_vs_index"
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="generated_question", columns=["generated_question", "statement_text", "manual_summarization", "table_list", "statement_id"]
    )
    return vectorstore.as_retriever()


# test our retriever
retriever = get_retriever()
similar_documents = retriever.get_relevant_documents("How do I track my Databricks Billing?")
print(f"Relevant documents: {similar_documents[0]}")

# COMMAND ----------

# MAGIC %md
# MAGIC Test the retrieer with filters.
# MAGIC
# MAGIC AFAIK, it's not possible to dynamically pass filters into the retriever, so here we use the retriever to get the vector store and perform our own similarity search.
# MAGIC
# MAGIC **Note: When performing a search with a filter, this is not an exclusive filter. Meaning that all it does is make sure that any of the tables in the list are any of the tables in the query. This means that the filter possibly could return queries that use tables outside of the provided list...and hence it's possible the LLM will use that additional information and generate SQL that's outside of the tables provided.**

# COMMAND ----------

# DBTITLE 1,Similarity Search with Filter
retriever.vectorstore.similarity_search('test', filters={"table_list": ['mfg_sales_demo.northwind.regions', 'mfg_sales_demo.northwind.orders']})

# COMMAND ----------

# DBTITLE 1,Generic Endpoint Function
import json, requests
import pandas as pd

#Define function for Endpoint Call
def call_endpoint(input_dict, endpoint_name, ):
  workspaceUrl = os.environ['DATABRICKS_HOST']
  token = os.environ['DATABRICKS_TOKEN']

  df = pd.DataFrame(input_dict)
  data = {'dataframe_split': df.to_dict(orient='split')}
  data_json = json.dumps(data, allow_nan=True)

  url = f"{workspaceUrl}/serving-endpoints/{endpoint_name}/invocations"
  headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}

  response = requests.request(method='POST', headers=headers, url=url, data=data_json)
  if response.status_code != 200:
      raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

# COMMAND ----------

# MAGIC %md
# MAGIC In other examples we used the DBRX Instruct pay-per-token endpoint provided by Databricks for our inference. Now, we are using a fine tuned Completions model deployed through Provisioned Throughput. As such, we will no longer use the `ChatDatabricks` library from `langchain_community.chat_models`. We will now use the `Databricks` library from `langchain_community.llms`. 
# MAGIC
# MAGIC Unfortunately, the library doesn't naturally handle the outputs from provisioned throughput models, so we must utilize the `transform_output_fn` feature to convert the endpoint response to a string. What this means is effectively navigating the json response to get the actual text inference and returning it.
# MAGIC
# MAGIC I have built the below function to work with a Llama3 endpoint, but if you use a different model, it may return inference in a different structure. If so, then modify to adjust for it.

# COMMAND ----------

# DBTITLE 1,Transform the Model Output to a String
def transform_output(response):
    return response["candidates"][0]["text"]

# COMMAND ----------

# DBTITLE 1,Build the Functions needed for the Chain

from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
import json, requests
import pandas as pd

from langchain_community.llms import Databricks
llama_ft_model = Databricks(endpoint_name="t2s_llama3_8b_ft", transform_output_fn=transform_output)

def mine_queries(queries):
    return {'sample_queries':[{'question':q.page_content, 'sql':q.metadata['statement_text']} for q in queries], 'tables':list(set([table for q in queries for table in q.metadata['table_list']]))}

def table_definitions(input):
    tables = list(set(input["tables"]))
    data = {'full_table': tables}
    resp = call_endpoint(data, "t2s_table_definition")

    input["table_response"] = resp['outputs']
    return input

def build_prompt(input):
    enriched = input["enriched_data"]
    question = input["prompt"]
    tables = enriched["table_response"]
    definitions = [t["table_definition"] for t in tables]
    samples = enriched["sample_queries"]

    data = {'question': [question], 'table_definitions': [definitions], 'sample_queries': [samples]}
    resp = call_endpoint(data, "t2s_build_prompt")

    input["full_prompt"] = resp['outputs'][0]['prompt']
    #print(input)
    return input

prompt = PromptTemplate.from_template("{full_prompt}")
output_parser = StrOutputParser()

# COMMAND ----------

# MAGIC %md
# MAGIC Introducing our new function to perform our own retrieval instead of using the retriever.

# COMMAND ----------

# DBTITLE 1,Retriever function for filters
def get_matches(input):
  query = input["prompt"]
  schema_filter = input.get("schema_filter", None) 
  table_filter = input.get("table_filter", None)
  filter = {}
  if schema_filter:
    filter["schema_list"] = schema_filter
  elif table_filter:
    if not isinstance(table_filter, list):
      table_filter = [table_filter]
    filter["table_list"] = list(set(table_filter))

  if filter:
    docs = retriever.vectorstore.similarity_search(query, filters=filter)
  else:
    docs = retriever.vectorstore.similarity_search(query)
    
  samples = [{'question':q.page_content, 'sql':q.metadata['statement_text']} for q in docs]
  if "table_list" in filter:
    tables = filter["table_list"]
  else:
    tables = list(set([table for q in docs for table in q.metadata['table_list']]))

  return {'sample_queries': samples, 'tables': tables}


# COMMAND ----------

# DBTITLE 1,Create Chain
setup_and_retrieval = RunnablePassthrough.assign(enriched_data = RunnableLambda(get_matches) | table_definitions)

chain = setup_and_retrieval | build_prompt | prompt | llama_ft_model | output_parser

# COMMAND ----------

# MAGIC %md
# MAGIC Test the result:

# COMMAND ----------

# DBTITLE 1,Test the Chain
import mlflow
#mlflow.langchain.autolog(log_models=True, log_input_examples=True) #uncomment to enable tracing

question = {"prompt":"who ordered the most items?", "schema_filter":["mfg_sales_demo.northwind"]}
answer = chain.invoke(question)
question_signature = {"prompt":"who ordered the most items?", "schema_filter":["mfg_sales_demo.northwind" ,None], "table_filter":["mfg_sales_demo.northwind" ,None]}
print(answer)

# COMMAND ----------

# MAGIC %md
# MAGIC Register the chain using the langchain flavor of mlflow.

# COMMAND ----------

# DBTITLE 1,Register the Chain to MLFlow
from mlflow.models import infer_signature
import mlflow
import langchain_core
import langchain_community

mlflow.set_registry_uri("databricks-uc")
model_name = f"{catalog}.text_to_sql.llama3_8b_ft_chain"

with mlflow.start_run(run_name="text_to_sql") as run:
    signature = infer_signature(question_signature, answer)
    model_info = mlflow.langchain.log_model(
        chain,
        loader_fn=get_retriever,  # Load the retriever with DATABRICKS_TOKEN env as secret (for authentication).
        artifact_path="chain",
        registered_model_name=model_name,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain-core==" + langchain_core.__version__,
            "langchain-community==" + langchain_community.__version__,
            "langchain-openai",
            "databricks-vectorsearch",
        ],
        input_example=question,
        signature=signature
    )

# COMMAND ----------

# DBTITLE 1,Test Model by Loading it and running it
model = mlflow.langchain.load_model(model_info.model_uri)
model.invoke(question)

# COMMAND ----------

# MAGIC %md
# MAGIC Now we will use model serving to deploy the rag chain on an endpoint.
# MAGIC
# MAGIC *Note: it often times out after 20 minutes but continues to spin up in the background. Check the Serving page to see when it is deployed.*

# COMMAND ----------

# DBTITLE 1,Serve the model
# Create or update serving endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput, ServedModelInputWorkloadSize

serving_endpoint_name = "t2s_llama3_8b_ft_chain"
latest_model_version = model_info.registered_model_version

w = WorkspaceClient()
endpoint_config = EndpointCoreConfigInput(
    name=serving_endpoint_name,
    served_models=[
        ServedModelInput(
            model_name=model_name,
            model_version=latest_model_version,
            workload_size=ServedModelInputWorkloadSize.SMALL,
            scale_to_zero_enabled=True,
            environment_vars={
                "DATABRICKS_HOST": "{{secrets/agent_studio/databricks_host}}",
                "DATABRICKS_TOKEN": "{{secrets/robert_mosley/databricks_token}}"# <scope>/<secret> that contains an access token
            }
        )
    ]
)

existing_endpoint = next(
    (e for e in w.serving_endpoints.list() if e.name == serving_endpoint_name), None
)
serving_endpoint_url = f"{workspaceUrl}/ml/endpoints/{serving_endpoint_name}"
if existing_endpoint == None:
    print(f"Creating the endpoint {serving_endpoint_url}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.create_and_wait(name=serving_endpoint_name, config=endpoint_config)
else:
    print(f"Updating the endpoint {serving_endpoint_url} to version {latest_model_version}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.update_config_and_wait(served_models=endpoint_config.served_models, name=serving_endpoint_name)
    
displayHTML(f'Your Model Endpoint Serving is now available. Open the <a href="/ml/endpoints/{serving_endpoint_name}">Model Serving Endpoint page</a> for more details.')
