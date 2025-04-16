import polars as pl
from main import vqa, extract_brace
from loguru import logger
logger.add('zero_bench.log')

df = pl.read_parquet('/data/zerobench-00000-of-00001.parquet')

# print(df.head())

keep_only = [1, 43, 77, 11, 44, 79, 13, 50, 80, 2, 5, 8, 57, 81, 22, 60, 82, 25, 6, 84, 26, 61, 86, 27, 63, 87, 29, 64, 89, 35, 65, 89, 37, 65, 90, 38, 67, 91, 39, 70, 92, 40, 7, 95, 4, 72, 97, 42, 76, 98]

for row in df.iter_rows(named=True):
    if int(row['question_id']) in keep_only and len(row['question_images']) == 1:
        response = vqa(row['question_images_decoded'][0]['bytes'], row['question_text'])
        if (extract_brace(response) or response == row['question_answer']):
            print('EUREKA!', row['question_id'])
            exit()
        logger.info(f'{row["question_id"]} \n\nQ: {row["question_text"]} \n\nA: {row["question_answer"]} \n\nR: {response}')