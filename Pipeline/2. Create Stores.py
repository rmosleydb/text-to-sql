# Databricks notebook source
# MAGIC %md
# MAGIC Here we build all the online serving layers that we will need for our text-to-sql solution. 
# MAGIC
# MAGIC This includes:
# MAGIC - Online Tables (and Functions): https://docs.databricks.com/en/machine-learning/feature-store/online-tables.html
# MAGIC - Vector Search: https://docs.databricks.com/en/generative-ai/vector-search.html

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %pip install databricks-vectorsearch
# MAGIC %pip install databricks-sdk --upgrade
# MAGIC %pip install mlflow>=2.9.0
# MAGIC %pip install databricks.feature_engineering
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Get Token and Workspace URL
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
workspaceUrl = spark.conf.get('spark.databricks.workspaceUrl')
workspaceUrl = f"https://{workspaceUrl}"

catalog = "robert_mosley"

# COMMAND ----------

# MAGIC %md
# MAGIC Start by creating the vector search for the query history.

# COMMAND ----------

# DBTITLE 1,Create Vector Search for Query History
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import *
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

#The table we'd like to index
source_table_fullname = "robert_mosley.text_to_sql.query_history_pre_index"
# Where we want to store our index
vs_index_fullname = "robert_mosley.text_to_sql.query_history_vs_index"
#compute endpoint
endpoint_name = "one-env-shared-endpoint-7"

#vsc.delete_index(endpoint_name, vs_index_fullname)
if True: #not index_exists(vsc, endpoint_name, vs_index_fullname):
  print(f"Creating index {vs_index_fullname} on endpoint {endpoint_name}...")
  vsc.create_delta_sync_index(
    endpoint_name=endpoint_name,
    index_name=vs_index_fullname,
    source_table_name=source_table_fullname,
    pipeline_type="TRIGGERED",
    primary_key="statement_id",
    embedding_source_column='generated_question', #The column containing our text
    embedding_model_endpoint_name='databricks-bge-large-en' #The embedding endpoint used to create the embeddings
  )
else:
  #Trigger a sync to update our vs content with the new data saved in the table
  wait_for_index_to_be_ready(vsc, endpoint_name, vs_index_fullname)
  vsc.get_index(endpoint_name, vs_index_fullname).sync()

print(f"index {vs_index_fullname} on table {source_table_fullname} is ready")

# COMMAND ----------

# MAGIC %md
# MAGIC Here, we will build out our first online store from the table_search_pre_index table we built in the last notebook. 
# MAGIC
# MAGIC If the store already exists, and you wish to update it, uncomment the delete command.

# COMMAND ----------

# DBTITLE 1,Create Table Online Store

from pprint import pprint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import *

w = WorkspaceClient(host=workspaceUrl, token=databricks_token)

# Create an online table
spec = OnlineTableSpec(
  primary_key_columns=["full_table"],
  source_table_full_name=f"{catalog}.text_to_sql.table_search_pre_index",
  run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({'triggered': 'true'})
)
#w.online_tables.delete(name=f"{catalog}.text_to_sql.table_search_os")
w.online_tables.create(name=f"{catalog}.text_to_sql.table_search_os", spec=spec)

# COMMAND ----------

# MAGIC %md
# MAGIC This feature spec receives a list of tables (catalog.schema.table) and outputs a list of DDL table definitions. It uses an online table lookup to get information needed for the python function to format. The formatted response is output by the function.
# MAGIC
# MAGIC After this, we build the endpoint that can be hit by the AI System.

# COMMAND ----------

# DBTITLE 1,Create Table Definition Online Function
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup, FeatureFunction

fe = FeatureEngineeringClient()
fe.delete_feature_spec(name="robert_mosley.text_to_sql.formatted_table_definition_fs")
fe.create_feature_spec(
  name="robert_mosley.text_to_sql.formatted_table_definition_fs",
  features=[
    FeatureLookup(
      table_name="robert_mosley.text_to_sql.table_search_pre_index",
      lookup_key=["full_table"]
    ),
    FeatureFunction(
      udf_name="robert_mosley.text_to_sql.format_table_definitions", 
      output_name="table_definition",
      input_bindings={
        "full_table": "full_table", 
        "comment": "comment", 
        "column_description_list": "column_description_list"
      },
    ),
  ]
)


# COMMAND ----------

# DBTITLE 1,Create Table Definition Endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

workspace = WorkspaceClient()
# Model Serving endpoint
endpoint_name = "t2s_table_definition"

workspace.serving_endpoints.delete(endpoint_name)

workspace.serving_endpoints.create(
  name=endpoint_name,
  config=EndpointCoreConfigInput(
    served_entities=[
      ServedEntityInput(
        entity_name=f"{catalog}.text_to_sql.formatted_table_definition_fs",
        scale_to_zero_enabled=True,
        workload_size="Small"
      )
    ]
  )
)

# COMMAND ----------

# MAGIC %md
# MAGIC This feature spec receives a prompt, a list of table definitions (built in the last endpoint), and a list of sample queries (identified from the vector search which we will build in a minute) and outputs a string that represents the prompt for the model.
# MAGIC
# MAGIC **Note: This is technically optional, but recommended. You could build the prompt using a Prompt Template (or similar means) in your app, but we will reuse the background function when fine tuning a model. This ensures that the exact same template is used for fine tuning that's used in the prompting.**
# MAGIC
# MAGIC After this, we build the endpoint that can be hit by the AI System.

# COMMAND ----------

# DBTITLE 1,Create Prompt Online Function
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup, FeatureFunction

fe = FeatureEngineeringClient()
fe.delete_feature_spec(name=f"{catalog}.text_to_sql.build_prompt_fs")
fe.create_feature_spec(
  name=f"{catalog}.text_to_sql.build_prompt_fs",
  features=[
    FeatureFunction(
      udf_name=f"{catalog}.text_to_sql.build_prompt", 
      output_name="prompt",
      input_bindings={
        "question": "question", 
        "table_definitions": "table_definitions", 
        "sample_queries": "sample_queries"
      },
    ),
  ]
)


# COMMAND ----------

# DBTITLE 1,Create Prompt Endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

workspace = WorkspaceClient()
# Model Serving endpoint
endpoint_name = "t2s_build_prompt"

workspace.serving_endpoints.delete(endpoint_name)
workspace.serving_endpoints.create(
  name=endpoint_name,
  config=EndpointCoreConfigInput(
    served_entities=[
      ServedEntityInput(
        entity_name=f"{catalog}.text_to_sql.build_prompt_fs",
        scale_to_zero_enabled=True,
        workload_size="Small"
      )
    ]
  )
)

# COMMAND ----------

# MAGIC %md
# MAGIC **This is optional.** It's designed to be a column lookup for an agent solution. This will be ideal in situations where customers have tables with lots (hundreds or thousands) of columns. The model solution should be engineered to lookup a list of columns and include them in the prompting design.
# MAGIC
# MAGIC This step:
# MAGIC - Creates an online store
# MAGIC - Creates a features spec definition to use the online store
# MAGIC - Creates an endpoint to support the lookup.

# COMMAND ----------

# DBTITLE 1,Create Column Online Store

from pprint import pprint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import *

w = WorkspaceClient(host=workspaceUrl, token=databricks_token)

# Create an online table
spec = OnlineTableSpec(
  primary_key_columns=["id"],
  source_table_full_name=f"{catalog}.text_to_sql.column_search_pre_index",
  run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({'triggered': 'true'})
)
#w.online_tables.delete(name='robert_mosley.text_to_sql.column_search_os')
w.online_tables.create(name=f"{catalog}.text_to_sql.column_search_os", spec=spec)

# COMMAND ----------

# DBTITLE 1,Create Column Feature Spec
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

fe = FeatureEngineeringClient()
fe.create_feature_spec(
  name="robert_mosley.text_to_sql.column_search_fs",
  features=[
    FeatureLookup(
      table_name="robert_mosley.text_to_sql.column_search_pre_index",
      lookup_key=["id"]
    )
  ]
)

# COMMAND ----------

# DBTITLE 1,Create Column Lookup Endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

workspace = WorkspaceClient()
# Model Serving endpoint
endpoint_name = "rm_t2s_column_lookup"

workspace.serving_endpoints.create(
  name=endpoint_name,
  config=EndpointCoreConfigInput(
    served_entities=[
      ServedEntityInput(
        entity_name="robert_mosley.text_to_sql.column_search_fs",
        scale_to_zero_enabled=True,
        workload_size="Small"
      )
    ]
  )
)

# COMMAND ----------

# MAGIC %md
# MAGIC **This is optional.** These are vector stores for Table and Column tables created in the first notebook. These will enable search on their searching columns, allowing an agent to dynamically lookup tables/columns based on similarity search.

# COMMAND ----------

# DBTITLE 1,Create Table Vector Search
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import *
from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient()

#The table we'd like to index
source_table_fullname = "robert_mosley.text_to_sql.table_search_pre_index"
# Where we want to store our index
vs_index_fullname = "robert_mosley.text_to_sql.table_search_vs_index"
#compute endpoint
endpoint_name = "one-env-shared-endpoint-7"

if True: #not index_exists(vsc, endpoint_name, vs_index_fullname):
  print(f"Creating index {vs_index_fullname} on endpoint {endpoint_name}...")
  vsc.create_delta_sync_index(
    endpoint_name=endpoint_name,
    index_name=vs_index_fullname,
    source_table_name=source_table_fullname,
    pipeline_type="TRIGGERED",
    primary_key="full_table",
    embedding_source_column='table_search', #The column containing our text
    embedding_model_endpoint_name='databricks-bge-large-en' #The embedding endpoint used to create the embeddings
  )
else:
  #Trigger a sync to update our vs content with the new data saved in the table
  wait_for_index_to_be_ready(vsc, endpoint_name, vs_index_fullname)
  vsc.get_index(endpoint_name, vs_index_fullname).sync()

print(f"index {vs_index_fullname} on table {source_table_fullname} is ready")

# COMMAND ----------

# DBTITLE 1,Create Column Vector Search
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import *
from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient()

#The table we'd like to index
source_table_fullname = "robert_mosley.text_to_sql.column_search_pre_index"
# Where we want to store our index
vs_index_fullname = "robert_mosley.text_to_sql.column_search_vs_index"
#compute endpoint
endpoint_name = "one-env-shared-endpoint-7"

if True: #not index_exists(vsc, endpoint_name, vs_index_fullname):
  print(f"Creating index {vs_index_fullname} on endpoint {endpoint_name}...")
  vsc.create_delta_sync_index(
    endpoint_name=endpoint_name,
    index_name=vs_index_fullname,
    source_table_name=source_table_fullname,
    pipeline_type="TRIGGERED",
    primary_key="id",
    embedding_source_column='column_search', #The column containing our text
    embedding_model_endpoint_name='databricks-bge-large-en' #The embedding endpoint used to create the embeddings
  )
else:
  #Trigger a sync to update our vs content with the new data saved in the table
  wait_for_index_to_be_ready(vsc, endpoint_name, vs_index_fullname)
  vsc.get_index(endpoint_name, vs_index_fullname).sync()

print(f"index {vs_index_fullname} on table {source_table_fullname} is ready")

# COMMAND ----------

# MAGIC %md
# MAGIC **This is optional.** It's designed to be a table lookup. Up above, we have used it in combination with a function to generate table definitions. This can be used if you need a separate lookup for tables outside of table definition build outs.
# MAGIC
# MAGIC This step:
# MAGIC - Reuses the above online store
# MAGIC - Creates a features spec definition to use the online store
# MAGIC - Creates an endpoint to support the lookup.

# COMMAND ----------

# DBTITLE 1,Create Table Feature Spec
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

fe = FeatureEngineeringClient()
#fe.delete_feature_spec(name=f"{catalog}.text_to_sql.table_search_fs")
fe.create_feature_spec(
  name=f"{catalog}text_to_sql.table_search_fs",
  features=[
    FeatureLookup(
      table_name=f"{catalog}.text_to_sql.table_search_pre_index",
      lookup_key=["full_table"]
    )
  ]
)

# COMMAND ----------

# DBTITLE 1,Create Table Lookup Endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

workspace = WorkspaceClient()
# Model Serving endpoint
endpoint_name = "t2s_table_lookup"

#workspace.serving_endpoints.delete(endpoint_name)

workspace.serving_endpoints.create(
  name=endpoint_name,
  config=EndpointCoreConfigInput(
    served_entities=[
      ServedEntityInput(
        entity_name=f"{catalog}.text_to_sql.table_search_fs",
        scale_to_zero_enabled=True,
        workload_size="Small"
      )
    ]
  )
)
