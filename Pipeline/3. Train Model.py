# Databricks notebook source
# MAGIC %md
# MAGIC We must create our training data set. To do this, we will use our query history table built in the first notebook. But to build a full prompt, we will need table definitions and similar queries. 

# COMMAND ----------

# MAGIC %md
# MAGIC We create a function for similarity search. Instead of recomputing the embeddings for each row each time, we use the precomupted embeddings from the table to identify the similarity between queries.

# COMMAND ----------

# MAGIC %sql
# MAGIC use catalog robert_mosley;

# COMMAND ----------

catalog = 'robert_mosley'

# COMMAND ----------

# DBTITLE 1,Create Function for Similarity Matching
# MAGIC %sql
# MAGIC CREATE or REPLACE FUNCTION text_to_sql.embedding_similarity_score(vec1 ARRAY<FLOAT>, vec2 ARRAY<FLOAT>)
# MAGIC   RETURNS FLOAT
# MAGIC   LANGUAGE PYTHON
# MAGIC   AS $$
# MAGIC     import math
# MAGIC     import numpy
# MAGIC     from scipy import spatial
# MAGIC
# MAGIC     vec1 = numpy.array(vec1, dtype=numpy.float32)
# MAGIC     vec2 = numpy.array(vec2, dtype=numpy.float32)
# MAGIC
# MAGIC     # ---- Easy Solution ----
# MAGIC     cosine_sim = spatial.distance.cosine(vec1, vec2)  
# MAGIC   
# MAGIC     return 1-cosine_sim
# MAGIC   $$

# COMMAND ----------

# MAGIC %sql
# MAGIC select array_size(ai_query("databricks-bge-large-en",generated_question, "ARRAY<FLOAT>")) size
# MAGIC   ,*
# MAGIC from text_to_sql.query_history_pre_index
# MAGIC where array_size(question_embeddings) <> 1024

# COMMAND ----------

# DBTITLE 1,See the similarity function work
# MAGIC %sql
# MAGIC select text_to_sql.embedding_similarity_score(
# MAGIC   ai_query("databricks-bge-large-en", "select * from table", "ARRAY<FLOAT>")
# MAGIC   , ai_query("databricks-bge-large-en", "select table.id from table", "ARRAY<FLOAT>")
# MAGIC   ) as score

# COMMAND ----------

# MAGIC %md
# MAGIC The following two blocks identify the top 3 similar queries per query. The requirement is that they must have parsed tables and must share a table between them.

# COMMAND ----------

# DBTITLE 1,Create Temporary View of query comparisons based on questions
# MAGIC %sql
# MAGIC create or replace temporary view query_compare as
# MAGIC with queries as (
# MAGIC select statement_id
# MAGIC   , statement_text
# MAGIC   , generated_question
# MAGIC   , manual_summarization
# MAGIC   , question_embeddings
# MAGIC   , statement_embeddings
# MAGIC   , table_list
# MAGIC from robert_mosley.text_to_sql.query_history_pre_index
# MAGIC where array_size(table_list) > 0
# MAGIC   and array_size(question_embeddings) = 1024
# MAGIC )
# MAGIC select a.statement_id as first_statement_id
# MAGIC   , b.statement_id as second_statement_id
# MAGIC   , text_to_sql.embedding_similarity_score(a.question_embeddings, b.question_embeddings) as question_similarity_score
# MAGIC   --, text_to_sql.embedding_similarity_score(a.statement_embeddings, b.statement_embeddings) as statement_similarity_score
# MAGIC from queries a, queries b 
# MAGIC where a.statement_id <> b.statement_id 
# MAGIC   and array_size(array_intersect(a.table_list, b.table_list)) > 0
# MAGIC

# COMMAND ----------

# DBTITLE 1,Build Table with top 3 matches
# MAGIC %sql
# MAGIC create or replace table text_to_sql.query_top_matches as
# MAGIC with ds as (
# MAGIC select first_statement_id as statement_id
# MAGIC   , second_statement_id as compared_statement_id
# MAGIC   , question_similarity_score
# MAGIC   , rank() over (partition by first_statement_id order by question_similarity_score desc) as question_similarity_rank
# MAGIC from query_compare c
# MAGIC )
# MAGIC   select statement_id
# MAGIC     , struct(compared_statement_id, question_similarity_score) as top_compared_questions
# MAGIC     , question_similarity_rank
# MAGIC   from ds
# MAGIC   where question_similarity_rank <= 3
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Take the results from above and build out a final dataset, matching up the 3 queries, pulling out the parsed tables, and reusing the functions from the first notebook to generate the final prompt. The resulting table will have a prompt/response format.

# COMMAND ----------

# MAGIC %sql
# MAGIC create or replace table text_to_sql.fine_tuning_set as
# MAGIC with training_set as (
# MAGIC   select m.statement_id, q2.statement_text
# MAGIC     , array_agg(struct(q.generated_question as question, q.statement_text as sql)) as sample_queries
# MAGIC     , array_compact(array_distinct(flatten(array_agg(q.table_list)))) as sample_table_list
# MAGIC     , q2.table_list
# MAGIC     , q2.generated_question
# MAGIC   from text_to_sql.query_top_matches m
# MAGIC     inner join text_to_sql.query_history_pre_index q on m.top_compared_questions.compared_statement_id = q.statement_id 
# MAGIC     inner join text_to_sql.query_history_pre_index q2 on m.statement_id = q2.statement_id 
# MAGIC   group by all
# MAGIC )
# MAGIC , ts2 as (
# MAGIC   select statement_id
# MAGIC     , generated_question
# MAGIC     , statement_text
# MAGIC     , sample_queries
# MAGIC     , explode(sample_table_list) as full_table
# MAGIC   from training_set
# MAGIC )
# MAGIC , ts3 as (
# MAGIC   select ts2.* except (full_table)
# MAGIC     , text_to_sql.format_table_definitions(t.full_table, t.comment, t.column_description_list) as table_definition
# MAGIC   from ts2
# MAGIC     inner join text_to_sql.table_search_pre_index t on ts2.full_table = t.full_table
# MAGIC )
# MAGIC , ts4 as (
# MAGIC   select ts3.* except (table_definition)
# MAGIC     , array_agg(table_definition) as table_definitions
# MAGIC   from ts3
# MAGIC   group by all
# MAGIC )
# MAGIC select statement_id
# MAGIC   , text_to_sql.build_prompt(generated_question, table_definitions, sample_queries) as full_prompt
# MAGIC   , statement_text as response
# MAGIC from ts4
# MAGIC

# COMMAND ----------

# DBTITLE 1,View the Results
# MAGIC %sql
# MAGIC select * from text_to_sql.fine_tuning_set

# COMMAND ----------

# MAGIC %md
# MAGIC Now we perform the fine tuning job, using the Databricks Foundational Model Training API.

# COMMAND ----------

from databricks.model_training import foundation_model as fm

# COMMAND ----------

# MAGIC %md
# MAGIC Write the dataset to a jsonl file.

# COMMAND ----------

# MAGIC %sql
# MAGIC create volume if not exists text_to_sql.training_files;

# COMMAND ----------

file_path = f"/Volumes/{catalog}/text_to_sql/training_files/t2s-query-history.jsonl"
df = spark.sql(f"SELECT full_prompt as prompt, response as response from {catalog}.text_to_sql.fine_tuning_set")
df.toPandas().to_json(file_path, orient='records', lines=True)

# COMMAND ----------

# Reading the file and printing its contents
with open(file_path, 'r') as file:
    for i, line in enumerate(file):
        if i < 5:  # Print only the first 5 lines
            print(line)

# COMMAND ----------

# MAGIC %md
# MAGIC Below, we are fine tuning the Llama 3 8B model. You can change this to a different model based on the various foundational models supported by Databricks. Please be aware of various (Terms of Service) TOS when inputting synthetically generated data into a fine tuning job. Generally speaking, it's best practice to keep data within the same model family - ie. use Llama 3 models to improve Llama 3 models. 
# MAGIC
# MAGIC **This is a legal requirement, not a technical requirement. Disclaimer: I am not a lawyer, and this is not a replacement for legal advice. Please consult a proper legal advisor before deploying a solution like this.**

# COMMAND ----------

# DBTITLE 1,Fine tune the model
model = 'meta-llama/Meta-Llama-3-8B'
train_data_path = file_path
register_to = f"{catalog}.text_to_sql.t2s_llama3_8b"
training_duration = '10ep'
learning_rate = '5e-8'
eval_prompts = []


run = fm.create(
  model=model,
  train_data_path=train_data_path,
  register_to=register_to,
  training_duration=training_duration,
  #learning_rate=learning_rate,
  #eval_prompts=eval_prompts,
)
run

# COMMAND ----------

# Events for current run
fm.get_events(run)

# COMMAND ----------

fm.list(limit=3)

# COMMAND ----------

register_to = "robert_mosley.text_to_sql.llama3_8b_t2s"

# COMMAND ----------

# MAGIC %pip install databricks-sdk --upgrade
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table robert_mosley.text_to_sql.t2s_llama3_8b_ft_payload

# COMMAND ----------

# MAGIC %md
# MAGIC Deploy the fine tuned model to model serving.

# COMMAND ----------


from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput, EndpointCoreConfigInput, AutoCaptureConfigInput

serving_endpoint_name = "t2s_llama3_8b_ft"
w = WorkspaceClient()
endpoint_config = EndpointCoreConfigInput(
    name=serving_endpoint_name,
    served_entities=[
        ServedEntityInput(
            entity_name=register_to,
            entity_version=1,
            min_provisioned_throughput=0, # The minimum tokens per second that the endpoint can scale down to.
            max_provisioned_throughput=100,# The maximum tokens per second that the endpoint can scale up to.
            scale_to_zero_enabled=True
        )
    ],
    auto_capture_config = AutoCaptureConfigInput(catalog_name=catalog, schema_name="text_to_sql", enabled=True)
)

force_update = False #Set this to True to release a newer version (the demo won't update the endpoint to a newer model version by default)
existing_endpoint = next(
    (e for e in w.serving_endpoints.list() if e.name == serving_endpoint_name), None
)
if existing_endpoint == None:
    print(f"Creating the endpoint {serving_endpoint_name}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.create_and_wait(name=serving_endpoint_name, config=endpoint_config)
else:
  print(f"endpoint {serving_endpoint_name} already exist...")
  if force_update:
    w.serving_endpoints.update_config_and_wait(served_entities=endpoint_config.served_entities, name=serving_endpoint_name)
