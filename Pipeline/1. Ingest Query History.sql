-- Databricks notebook source
-- MAGIC %md
-- MAGIC
-- MAGIC # Unity Catalog Data Ingestion Notebook
-- MAGIC This notebook provides a solution for ingesting table structures and query history from Unity Catalog into a format that can be consumed by LLMs for text-to-SQL applications. By leveraging the power of Unity Catalog metadata and LLMs, customers can enhance their data exploration and analysis capabilities.
-- MAGIC
-- MAGIC ## Overview
-- MAGIC The notebook focuses on two main tasks:
-- MAGIC
-- MAGIC 1. Extracting Table Structures: It retrieves the metadata of tables stored in Unity Catalog, including table names, column names, data types, and other relevant information. This information is crucial for understanding the structure of the data and forming accurate SQL queries.
-- MAGIC 2. Capturing Query History: It captures the historical queries executed on the tables in Unity Catalog. Then it uses AI Functions to extract table names used in each query. Then AI Functions are used to generate synthetic questions and summaries. This will be useful for both the RAG chain and Fine Tuning an LLM later on.
-- MAGIC
-- MAGIC <img src="https://github.com/rmosleydb/text-to-sql/blob/main/_resources/T2S_pipeline_1.png?raw=true" width="800">
-- MAGIC
-- MAGIC ## Conclusion
-- MAGIC By leveraging the power of Unity Catalog and Language and Learning Models (LLMs), this notebook provides a solution for ingesting table structures and query history into a format suitable for text-to-SQL applications. Customers can benefit from improved data exploration, enhanced query generation, and optimized query performance. Follow the usage steps to integrate this notebook into your data analysis workflow and unlock the potential of LLMs for seamless data interaction.

-- COMMAND ----------

-- DBTITLE 1,Setup Schema
use catalog robert_mosley;
create schema if not exists text_to_sql;

-- COMMAND ----------

-- DBTITLE 1,Build a minimum unique name
create or replace table text_to_sql.unique_table as 
with table_set as (
  select t.table_catalog
      , t.table_schema
      , concat(t.table_catalog, ".", t.table_schema, ".", t.table_name) as full_table_name
      , concat(t.table_schema, ".", t.table_name) as schema_table_name
      , t.table_name
      , count(1) over (partition by concat(t.table_schema, ".", t.table_name)) as schema_table_name_count
      , count(1) over (partition by t.table_name) as table_name_count
  from system.information_schema.tables t
)
select full_table_name
  , table_catalog
  , table_schema
  , table_name
  , case when table_name_count = 1 then 'TABLE'
        when schema_table_name_count = 1 then 'SCHEMA' else 'CATALOG' end as unique_level
  , case when table_name_count = 1 then 1
        when schema_table_name_count = 1 then 2 else 3 end as unique_level_nbr
  , case when table_name_count = 1 then table_name
        when schema_table_name_count = 1 then schema_table_name else full_table_name end as unique_table_name
from table_set

-- COMMAND ----------

-- DBTITLE 1,View Table Uniqueness by Namespace level
select unique_level
  , count(1) as table_cnt 
from text_to_sql.unique_table
group by all


-- COMMAND ----------

-- MAGIC %md
-- MAGIC Next we will get a unique set of queries from the query history.

-- COMMAND ----------

-- DBTITLE 1,Retrieve Unique set of Queries from History
create or replace table text_to_sql.query_history_unique
with cte as (
  select statement_id
    , REGEXP_REPLACE(REGEXP_REPLACE(statement_text, '\'([^\']+)\'', '\'{value}\''), '"([^"]+)"', '"{value}"') as statement_text
    , end_time
  from system.query.history
  --from text_to_sql.query_history_api
  where execution_status = 'FINISHED'
    and statement_type = 'SELECT'
)
select distinct first_value(statement_id) over (partition by statement_text order by end_time desc) as statement_id
  , statement_text
from cte

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Using similarity scores, we analyze subsequent queries from the same user and remove extremely similar queries to eliminate situations where someone may be iterating over a query. If the queries are similar, we take the most recently run version.
-- MAGIC

-- COMMAND ----------

-- DBTITLE 1,Perform a Similarity Check between Executions
create table text_to_sql.query_history_similarity_filter as 
with cte as (
  select h.* 
    , lead(h.end_time) over (partition by warehouse_id, executed_by order by end_time asc) as next_end_time
    , timestampdiff(MINUTE, end_time, lead(h.start_time) over (partition by warehouse_id, executed_by order by end_time asc))/60 as minutes_until_next_execution
    , lead(h.statement_text) over (partition by warehouse_id, executed_by order by end_time asc) next_statement_text
    , lead(h.statement_id) over (partition by warehouse_id, executed_by order by end_time asc) next_statement_id
  from system.query.history h
  --from text_to_sql.query_history_api h
    inner join text_to_sql.query_history_unique l on h.statement_id = l.statement_id
)
, similarity as (
  select *
    , ai_similarity(coalesce(next_statement_text, ''), statement_text) as next_similarity_score
  from cte
  order by end_time desc
  limit 10
)
select * from similarity
where next_similarity_score < .93 
order by executed_by, end_time desc
;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Now we begin to extract tables out of the sql query.

-- COMMAND ----------

create or replace function text_to_sql.pull_tables_from_json(json_str STRING)
    RETURNS ARRAY<STRING>
    LANGUAGE PYTHON
    as $$
        import json
        array_str = ""
        try:
            start_index = json_str.index('[')
            end_index = json_str.index(']') + 1
            array_str = json_str[start_index:end_index]
            array_str = array_str.replace("\_", "_").replace("`", "").lower().strip()
            ret = json.loads(array_str)
            
            return ret
        except:
            return ["PARSING ERROR", array_str]
    $$

-- COMMAND ----------

-- MAGIC %md
-- MAGIC We are going to fine tune a llama3 model in the 3rd notebook. Due to TOS of many models, as we generate synthetic data using models, we often need to generate the synthetic data within the same model family (ie, llama 3 70b to fine tune llama 3 8b). 
-- MAGIC
-- MAGIC Llama3 doesn't have a big enough context window to handle the largest queries, so we will use mixtral for those, which performs worse, but has a broader TOS when it comes to synthetic data being used to improve other models.
-- MAGIC
-- MAGIC Disclaimer: I am not a lawyer, and this shouldn't be taken as legal advice. Please consult with a legal representative before accepting the above claims.

-- COMMAND ----------

-- DBTITLE 1,Extract Tables from Longer SQL
create or replace table text_to_sql.table_extract as
with long_statements as (
  select *
  from text_to_sql.query_history_similarity_filter
  where len(statement_text) >= 20000
)
select statement_id
  , statement_text
  , ai_query("databricks-mixtral-8x7b-instruct", concat("Extract the table names out of the following SQL statement. Think carefully and do NOT confuse a reference to an alias or a CTE as a table name. Where applicable, extract the fully declared table name (catalog.schema.table). Output the tables, and only the tables, in a json code block. For example: \nExample SQL:\nwith cte as (\n select customer_id\n , name\n from customer\n)\nselect c.name\n , order_number\nfrom sap.sales.orders o \n inner join cte c on o.customer_id = c.customer_id\n\n```json\n{\n  \"tables\": [\"customer\", \"sap.sales.orders\"]\n}\n```\n\nRemember, only output the json code block, and think carefully about the tables you extract. If there are no tables, leave the array empty.\n\nSQL:\n", statement_text)) as manual_table_list
  , "databricks-mixtral-8x7b-instruct" as extraction_model
from long_statements


-- COMMAND ----------

-- DBTITLE 1,Extract Tables from Shorter SQL
--create or replace table text_to_sql.table_extract as
--databricks-mixtral-8x7b-instruct
--databricks-meta-llama-3-70b-instruct
insert into text_to_sql.table_extract
with short_statements as (
  select * 
  from text_to_sql.query_history_similarity_filter
  where statement_id not in (select statement_id from text_to_sql.table_extract)
)
select statement_id
  , statement_text
  , ai_query("databricks-meta-llama-3-70b-instruct", concat("Extract the table names out of the following SQL statement. Think carefully and do NOT confuse a reference to an alias or a CTE as a table name. Where applicable, extract the fully declared table name (catalog.schema.table). Output the tables, and only the tables, in a json code block. For example: \nExample SQL:\nwith cte as (\n select customer_id\n , name\n from customer\n)\nselect c.name\n , order_number\nfrom sap.sales.orders o \n inner join cte c on o.customer_id = c.customer_id\n\n```json\n{\n  \"tables\": [\"customer\", \"sap.sales.orders\"]\n}\n```\n\nRemember, only output the json code block, and think carefully about the tables you extract. If there are no tables, leave the array empty.\n\nSQL:\n", statement_text)) as manual_table_list
  , "databricks-meta-llama-3-70b-instruct" as extraction_model
from short_statements

-- COMMAND ----------

-- DBTITLE 1,Parse Table Names from JSON
create or replace table text_to_sql.table_extract_parse as 
select * 
  , text_to_sql.pull_tables_from_json(manual_table_list) as parsed_table_list
from text_to_sql.table_extract

-- COMMAND ----------

-- DBTITLE 1,Check Parsing Errors
select * from text_to_sql.table_extract_parse
where array_contains(parsed_table_list, "PARSING ERROR")

-- COMMAND ----------

-- DBTITLE 1,Retrieve Table Components
create or replace table text_to_sql.table_extract_final
with cte as (
  select explode(parsed_table_list) as table_contender
    , * except (parsed_table_list)
  from text_to_sql.table_extract_parse
)
, cte2 as (
select table_contender as full_table
  , element_at(split(table_contender, '\\.'), -1) as table_name
  , element_at(split(table_contender, '\\.'), -2)as schema_name
  , element_at(split(table_contender, '\\.'), -3) as catalog_name
  , * except (table_contender)
from cte)
, cte3 as (
select full_table
  , case when contains(statement_text, table_name) then table_name else null end as valid_table_name
  , case when contains(statement_text, schema_name) then schema_name else null end as valid_schema_name
  , case when contains(statement_text, catalog_name) then catalog_name else null end as valid_catalog_name
  , * except (full_table)
from cte2
)
, cte4 as (
  /* 
   - verifies that the table reference is unique enough to map back to a table in UC 
   - sets the catalog and schema name where it might be null from the parsing.*/
  select cte3.* except (valid_catalog_name, valid_schema_name)
    , coalesce(cte3.valid_catalog_name, u.table_catalog) as valid_catalog_name
    , coalesce(cte3.valid_schema_name, u.table_schema) as valid_schema_name
    , case when u.table_name is not null then true else false end as matched_ind
    , case when coalesce(cte3.valid_catalog_name, '') <> u.table_catalog or coalesce(cte3.valid_schema_name, '') <> u.table_schema then true else false end as fixed_ind
    , cte3.extraction_model
  from cte3
    left outer join text_to_sql.unique_table u on cte3.valid_table_name = u.table_name 
      and (cte3.valid_schema_name = u.table_schema or (u.unique_level_nbr = 1 and cte3.valid_schema_name is null))
      and (cte3.valid_catalog_name = u.table_catalog or (u.unique_level_nbr < 3 and cte3.valid_schema_name is null))
)
select statement_id
  , array_agg(struct(valid_catalog_name as catalog_name, valid_schema_name as schema_name, valid_table_name as table_name)) as table_struct_list
  , array_compact(array_distinct(array_agg(valid_catalog_name))) as catalog_list
  , array_compact(array_distinct(array_agg(concat(valid_catalog_name, '.', valid_schema_name)))) as schema_list
  , array_compact(array_distinct(array_agg(concat(valid_catalog_name, '.', valid_schema_name, '.', valid_table_name)))) as table_list
  , statement_text
  , array_contains(array_agg(matched_ind), true) as matched_ind
  , array_contains(array_agg(fixed_ind), true) as fixed_ind
  , cte4.extraction_model
from cte4
where valid_table_name is not null
  -- filter out tables where there's no unique tie to the current UC table list
  and valid_catalog_name is not null 
  and valid_schema_name is not null 
group by all


-- COMMAND ----------

-- MAGIC %md
-- MAGIC Now we generate synthetic data for SQL Queries: 
-- MAGIC 1. summaries (unused for now, but they could be useful later)
-- MAGIC 2. sample questions

-- COMMAND ----------

-- DBTITLE 1,Summarize Query and Generate Question
create or replace table text_to_sql.summary_processed as
select q.statement_id
, q.executed_by
, q.next_similarity_score
, ai_query("databricks-meta-llama-3-70b-instruct", concat("summarize what the following SQL is doing in plain english, in 100 words or less: \n", q.statement_text)) as manual_summarization
--, ai_summarize(statement_text, 100) as sql_summarization
, trim('"' from ai_query("databricks-meta-llama-3-70b-instruct", concat("Reverse Engineer Text-to-SQL: Write a user question or request that would result in this query, in 50 words or less: \n", q.statement_text))) as generated_question
, q.statement_text
, t.extraction_model
--, q.next_statement_text
--, q.start_time
--, q.end_time
from text_to_sql.query_history_similarity_filter q
  -- filter to statements that can be matched to an existing table
  inner join text_to_sql.table_extract_final t on q.statement_id = t.statement_id and t.matched_ind
where t.extraction_model = "databricks-meta-llama-3-70b-instruct"
order by executed_by, end_time desc

-- COMMAND ----------

-- DBTITLE 1,Round 2 (Mixtral Generation)
insert into text_to_sql.summary_processed
select q.statement_id
, q.executed_by
, q.next_similarity_score
, ai_query("databricks-mixtral-8x7b-instruct", concat("summarize what the following SQL is doing in plain english, in 100 words or less: \n", q.statement_text)) as manual_summarization
--, ai_summarize(statement_text, 100) as sql_summarization
, trim('"' from ai_query("databricks-mixtral-8x7b-instruct", concat("Reverse Engineer Text-to-SQL: Write a user question or request that would result in this query, in 50 words or less: \n", q.statement_text))) as generated_question
, q.statement_text
, t.extraction_model
--, q.next_statement_text
--, q.start_time
--, q.end_time
from text_to_sql.query_history_similarity_filter q
  -- filter to statements that can be matched to an existing table
  inner join text_to_sql.table_extract_final t on q.statement_id = t.statement_id and t.matched_ind
where t.extraction_model = "databricks-mixtral-8x7b-instruct"
order by executed_by, end_time desc

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Now, we generate Embeddings of questions and statements. We will use the question embeddings for the fine tuning notebook (#3)

-- COMMAND ----------

-- DBTITLE 1,Generate Embeddings for Questions and Statements
create or replace table robert_mosley.text_to_sql.query_history_embeddings as
select p.statement_id
  , p.generated_question
  , p.statement_text
  , ai_query("databricks-bge-large-en", p.generated_question, "ARRAY<FLOAT>") as question_embeddings
  , ai_query("databricks-bge-large-en", p.statement_text, "ARRAY<FLOAT>") as statement_embeddings
from robert_mosley.text_to_sql.summary_processed p

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Now we create our unified table that will be used for vector search and other tasks.

-- COMMAND ----------

-- DBTITLE 1,Bring Everything Together
create or replace table robert_mosley.text_to_sql.query_history_pre_index as
select p.* except (next_statement_text)
  --, t.table_struct_list
  , t.catalog_list
  , t.schema_list
  , t.table_list
  , e.question_embeddings
  , e.statement_embeddings
from robert_mosley.text_to_sql.summary_processed p
  inner join robert_mosley.text_to_sql.query_history_embeddings e on p.statement_id = e.statement_id
  left outer join robert_mosley.text_to_sql.table_extract_final t on p.statement_id = t.statement_id

-- COMMAND ----------

-- MAGIC %md
-- MAGIC We also create table and column datasets that can be used for the model chain.

-- COMMAND ----------

-- DBTITLE 1,Create a Table of Tables
create or replace table table_search_pre_index as
select t.table_catalog
  , t.table_schema
  , t.table_name
  , array_join(array(t.table_catalog, t.table_schema, t.table_name), '.') as full_table
  , t.table_type
  , t.comment
  , array_join(array_agg(c.column_name), ', ') as column_list
  , array_join(array_agg(concat("  ", c.column_name, " ", upper(c.full_data_type), case when c.comment is not null then concat(" COMMENT '", c.comment, "'") else "" end)), ',\n') column_description_list
  , to_json(named_struct('table', array_join(array(t.table_catalog, t.table_schema, t.table_name), '.'), 'comment', t.comment)) as table_search
from system.information_schema.tables t
  left outer join system.information_schema.columns c on t.table_catalog = c.table_catalog and t.table_schema = c.table_schema and t.table_name = c.table_name
group by t.table_catalog, t.table_schema, t.table_name, t.table_type, t.comment

-- COMMAND ----------

-- DBTITLE 1,Create a Table of Columns
create or replace table column_search_pre_index as
select array_join(array(table_catalog, table_schema, table_name, column_name), '.') as id
  , array_join(array(table_catalog, table_schema, table_name), '.') as full_table
  , column_name
  , is_nullable
  , full_data_type as data_type
  , comment
  , to_json(named_struct('table', array_join(array(table_catalog, table_schema, table_name), '.'), 'column_name', column_name, 'comment', comment)) as column_search
from system.information_schema.columns


-- COMMAND ----------

-- DBTITLE 1,Enable Change Data Feed
ALTER TABLE `robert_mosley`.`text_to_sql`.`query_history_pre_index` SET TBLPROPERTIES (delta.enableChangeDataFeed = true);
ALTER TABLE `robert_mosley`.`text_to_sql`.`table_search_pre_index` SET TBLPROPERTIES (delta.enableChangeDataFeed = true);
ALTER TABLE `robert_mosley`.`text_to_sql`.`column_search_pre_index` SET TBLPROPERTIES (delta.enableChangeDataFeed = true);

-- COMMAND ----------

-- DBTITLE 1,Set Table Primary Keys
-- 1. Modify the column to be non-nullable
ALTER TABLE `robert_mosley`.`text_to_sql`.`table_search_pre_index`
alter COLUMN `full_table` set NOT NULL;

ALTER TABLE `robert_mosley`.`text_to_sql`.`table_search_pre_index` ADD CONSTRAINT table_search_pre_index_pk PRIMARY KEY(full_table);

-- COMMAND ----------

-- DBTITLE 1,Set Column Primary keys
-- 1. Modify the column to be non-nullable
ALTER TABLE `robert_mosley`.`text_to_sql`.`column_search_pre_index`
alter COLUMN `id` set NOT NULL;

ALTER TABLE `robert_mosley`.`text_to_sql`.`column_search_pre_index` ADD CONSTRAINT column_search_pre_index_pk_pk PRIMARY KEY(id);

-- COMMAND ----------

-- MAGIC %md
-- MAGIC We create functions that will be used in both the RAG chain and in the Model Training.

-- COMMAND ----------

-- DBTITLE 1,Create Function for Retrieving Table Definitions
CREATE OR REPLACE FUNCTION robert_mosley.text_to_sql.get_table_definitions(tables ARRAY<STRING>)
    RETURNS ARRAY<STRING>
    RETURN 
    WITH cte as (
    SELECT robert_mosley.text_to_sql.format_table_definitions(t.full_table, t.comment, t.column_description_list) as table_definitions
    FROM robert_mosley.text_to_sql.table_search_pre_index t
    WHERE array_contains(tables, t.full_table)
    )
    select array_agg(table_definitions) from cte;

-- COMMAND ----------

-- DBTITLE 1,Build Function for Online Functions
CREATE OR REPLACE FUNCTION robert_mosley.text_to_sql.format_table_definitions(full_table STRING, comment STRING, column_description_list STRING)
    RETURNS STRING
    LANGUAGE PYTHON
    as $$
        try:
            if not comment:
                comment = ''
            else:
                comment = "\nCOMMENT '" + comment + "'"

            ret = f"CREATE TABLE {full_table} (\n{column_description_list}){comment};"

            return ret
        except:
            return ""
    $$

-- COMMAND ----------

-- MAGIC %md
-- MAGIC **Special Note for build_prompt**
-- MAGIC
-- MAGIC This function serves as a prompt template. Langchain has prompt features that allow you to create a prompt from a prompt template, so you won't necessarily need this function for the RAG Chain (though my notebooks do use it).
-- MAGIC
-- MAGIC The reason for this is because I use this function in the fine tuning notebook (#3), and by relying on the same function in both processes, I guarantee that I will fine-tune my models *on the exact same prompt format* that I pass into the model as part of the RAG chain. This way I don't have to manage the prompt template in two different places.

-- COMMAND ----------

-- DBTITLE 1,Build function for prompt
CREATE or REPLACE FUNCTION robert_mosley.text_to_sql.build_prompt(question STRING, table_definitions ARRAY<STRING>, sample_queries ARRAY<STRUCT<question: STRING, sql: STRING>>)
  RETURNS STRING
  LANGUAGE PYTHON
  AS $$
    PROMPT_TEMPLATE = """--Instructions\n/*\nYou are a helpful assistant that replies with SQL queries. Users will provide you with table definitions and sample SQL queries for similar prompts. Then users will submit a prompt using a SQL comment. Answer the user's prompt with a SELECT SQL statement ending with a ;.\n*/
\n--Tables\n{tables}
\n--Sample Prompt and SQL Response\n{samples}
\n--User Prompt\n/*\n{question}\n*/
\n--Your SQL Response\n"""

    samples = '\n\n'.join([f"/*\n{q['question']}\n*/ \n{q['sql']}" for q in sample_queries])
    tables = '\n\n'.join(table_definitions)

    ret = PROMPT_TEMPLATE.format(question=question, tables=tables, samples=samples)

    return ret
  $$

-- COMMAND ----------

-- DBTITLE 1,Test Functions
select robert_mosley.text_to_sql.build_prompt('What is the average number of orders per product?', array('system.information_schema.columns', 'system.information_schema.tables'), array(named_struct("question",'points per game', "sql",'SELECT SUM(POINTS), game from scores group by game;'), named_struct("question",'points per game', "sql",'SELECT SUM(POINTS), game from scores group by game;')))
