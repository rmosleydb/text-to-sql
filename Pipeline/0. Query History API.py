# Databricks notebook source
# MAGIC %md
# MAGIC # System Table Workaround
# MAGIC
# MAGIC The `system.query.history` table is in private preview and unavailable to new customers until it goes into Public Preview - currently slated for mid-July 2024. To help organizations move forward, this is a notebook that downloads the query history from the Databricks API. 
# MAGIC
# MAGIC *If you end up using this approach, in the first notebook `1. Ingest Query History`, be sure to search for `system.query.history`. It will be in two different cells. Below each reference is a commented out line that uses the table we created below. Be sure to swap out the lines, and it should work seemlessly with the rest of the notebook.*

# COMMAND ----------

# MAGIC %sql
# MAGIC use catalog robert_mosley;
# MAGIC create schema if not exists text_to_sql;

# COMMAND ----------

databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
workspaceUrl = spark.conf.get('spark.databricks.workspaceUrl')
workspaceUrl = f"https://{workspaceUrl}"

catalog = "robert_mosley"
record_count = 10000

# COMMAND ----------

import requests
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from pyspark.sql.types import StructType, StructField, StringType, LongType

# Initialize Spark session
#spark = SparkSession.builder.appName("QueryHistory").getOrCreate()

# Databricks API details
DATABRICKS_INSTANCE = workspaceUrl  # Replace with your Databricks instance URL
API_TOKEN = databricks_token  # Replace with your Databricks API token
API_URL = f"{DATABRICKS_INSTANCE}/api/2.0/sql/history/queries"
workspace_id = '1444828305810485'

# Initialize an empty DataFrame with the desired schema
schema = StructType([
    StructField("account_id", StringType(), True),
    StructField("workspace_id", StringType(), True),
    StructField("statement_id", StringType(), True),
    StructField("executed_by", StringType(), True),
    StructField("session_id", StringType(), True),
    StructField("execution_status", StringType(), True),
    StructField("warehouse_id", StringType(), True),
    StructField("executed_by_user_id", LongType(), True),
    StructField("statement_text", StringType(), True),
    StructField("statement_type", StringType(), True),
    StructField("error_message", StringType(), True),
    StructField("warehouse_channel", StringType(), True),
    StructField("client_application", StringType(), True),
    StructField("start_time", LongType(), True),
    StructField("end_time", LongType(), True),
])
query_history_df = spark.createDataFrame([], schema)

# Function to fetch query history data
def fetch_query_history(page_token=None):
    headers = {
        "Authorization": f"Bearer {API_TOKEN}"
    }
    params = {
        "max_results": 1000,
        "page_token": page_token
    }
    response = requests.get(API_URL, headers=headers, params=params)
    response.raise_for_status()
    return response.json()

# Fetch query history data iteratively
page_token = None
while True:
    result = fetch_query_history(page_token)
    queries = result.get("res", [])
    
    rows = []
    for query in queries:
        row = {
            "account_id": None,  # Placeholder, replace with actual account ID if available
            "workspace_id": workspace_id,  # Placeholder, replace with actual workspace ID if available
            "statement_id": query.get("query_id"),
            "executed_by": query.get("user_name"),
            "session_id": None,  # Placeholder, replace with actual session ID if available
            "execution_status": query.get("status"),
            "warehouse_id": query.get("warehouse_id"),
            "executed_by_user_id": query.get("user_id"),
            "statement_text": query.get("query_text"),
            "statement_type": query.get("statement_type"),
            "error_message": query.get("error_message"),
            "warehouse_channel": query.get("channel_used", {}).get("name"),
            "client_application": None,  # Placeholder, replace with actual client application if available
            "start_time": query.get("query_start_time_ms"),
            "end_time": query.get("execution_end_time_ms"),
        }
        rows.append(row)
    
    # Convert to DataFrame and append to existing DataFrame
    temp_df = spark.createDataFrame(rows, schema)
    query_history_df = query_history_df.union(temp_df)
    
    # Check if there's another page
    if not result.get("has_next_page") or query_history_df.count() >= record_count:
        break
    page_token = result.get("next_page_token")

# Show the DataFrame
display(query_history_df)

# COMMAND ----------

from pyspark.sql.functions import from_unixtime, col, floor

#convert unix integers to timestamps
query_history_df_time = query_history_df.withColumn(
    "start_time", 
    from_unixtime(floor(col("start_time")/1000))
).withColumn(
    "end_time", 
    from_unixtime(floor(col("end_time")/1000))
)

#write to table
query_history_df_time.write.mode("overwrite").saveAsTable(f"{catalog}.text_to_sql.query_history_api")
