# Databricks notebook source
# MAGIC %md
# MAGIC This is a basic RAG chain designed to receive a prompt and output a sql result.
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

# DBTITLE 1,Build the Functions and Chain

from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
import json, requests
import pandas as pd

from langchain_community.chat_models import ChatDatabricks
dbrx_model = ChatDatabricks(endpoint="databricks-dbrx-instruct")


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

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a Text-To-SQL generator. When prompted with a question, table structures and sample queries, only reply with SQL. No explanations. Avoid CTEs just to abstract the table name."),
        ("human", "{full_prompt}"),
    ]
)

output_parser = StrOutputParser()

setup_and_retrieval = RunnableParallel(
    {"enriched_data": retriever | mine_queries | table_definitions, "prompt": RunnablePassthrough()}
)
chain = setup_and_retrieval | build_prompt | chat_prompt | dbrx_model | output_parser

# COMMAND ----------

# MAGIC %md
# MAGIC Test the result:

# COMMAND ----------

# DBTITLE 1,Test the Chain
# langchain.debug = True #uncomment to see the chain details and the full prompt being sent
question = "How can I track billing usage on my workspaces?"
answer = chain.invoke(question)
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
model_name = f"{catalog}.text_to_sql.dbrx_chain"

with mlflow.start_run(run_name="text_to_sql") as run:
    signature = infer_signature(question, answer)
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
import mlflow
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

serving_endpoint_name = "t2s_dbrx"
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
