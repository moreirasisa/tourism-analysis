from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import StringType
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.sql.functions import round
from resources.resources import COUNTRY_LIST, COUNTRY_ABBREVIATIONS

# Caminhos dos arquivos
TURISMO_PATH = "src/dataset/csv/turismo_dataset.csv"
PARQUET_PATH = "src/dataset/parquet/Silver_Table"
CSV_PATH = "src/dataset/csv/final"

# Inicialização do Spark
spark = SparkSession.builder \
    .appName("CreateSilverTable") \
    .config("spark.driver.memory", "10g") \
    .config("spark.executor.memory", "10g") \
    .getOrCreate()

class CountryCleansing:
    def __init__(self):
        pass

    def train(self, df: DataFrame) -> DataFrame:
        # Cria dados de treinamento a partir de COUNTRY_ABBREVIATIONS
        train_data = spark.createDataFrame(
            [(country, abbr, 1) for abbr, country in COUNTRY_ABBREVIATIONS.items()],
            ["Country", "Abbreviation", "Match"]
        )

        # Pipeline de tokenização e regressão logística
        country_tokenizer = Tokenizer(inputCol="Country", outputCol="country_words")
        abbreviation_tokenizer = Tokenizer(inputCol="Abbreviation", outputCol="abbreviation_words")

        hashing_tf_country = HashingTF(inputCol="country_words", outputCol="country_tf")
        hashing_tf_abbreviation = HashingTF(inputCol="abbreviation_words", outputCol="abbreviation_tf")

        idf_country = IDF(inputCol="country_tf", outputCol="country_tfidf")
        idf_abbreviation = IDF(inputCol="abbreviation_tf", outputCol="abbreviation_tfidf")

        assembler = VectorAssembler(inputCols=["country_tfidf", "abbreviation_tfidf"], outputCol="features")
        lr = LogisticRegression(featuresCol="features", labelCol="Match")

        pipeline = Pipeline(stages=[
            country_tokenizer, abbreviation_tokenizer,
            hashing_tf_country, hashing_tf_abbreviation,
            idf_country, idf_abbreviation,
            assembler, lr
        ])

        model = pipeline.fit(train_data)

        prediction_df = model.transform(df)
        prediction_df = prediction_df.withColumn(
            "Country_Validated",
            F.when(F.col("prediction") == 1, F.col("Country"))
            .otherwise(F.lit("Not Matched"))
        )

        df = prediction_df.drop("features", "country_words", "abbreviation_words", "country_tf", 
                                "abbreviation_tf", "country_tfidf", "abbreviation_tfidf", 
                                "rawPrediction", "probability", "prediction")
        
        return df

class TourismTreatment:
    def __init__(self):
        self.country_list = COUNTRY_LIST

    def get_table(self, table_path: str) -> DataFrame:
        return spark.read.csv(table_path, header=True, inferSchema=True)
    
    def treatment_process(self, df: DataFrame) -> DataFrame:
        df = (
            df.withColumn('AverageTicketPrice', F.round(F.expr("float(split(`Combined Info`, ' ')[0])"), 2))
              .withColumn('Visitors', F.round(F.expr("float(replace(split(`Combined Info`, ' ')[3], 'M', ''))"), 2))
              .withColumn("Country", F.upper(F.col("Country")))
              .withColumn("Month", F.upper(F.col("Month")))
              .drop('Combined Info')
        )

        df = df.withColumn("ReversedCountry", F.reverse(F.col("Country")))

        df = df.withColumn(
            "Country",
            F.when(F.col("ReversedCountry").isin(self.country_list), F.col("ReversedCountry")).otherwise(F.col("Country"))
        ).drop("ReversedCountry")

        abbreviations_df = spark.createDataFrame(
            [(k, v) for k, v in COUNTRY_ABBREVIATIONS.items()],
            ["Abbreviation", "FullCountryName"]
        )
        
        df = df.join(abbreviations_df, df["Country"] == abbreviations_df["Abbreviation"], "left") \
               .withColumn("Country", F.coalesce(F.col("FullCountryName"), F.col("Country"))) \
               .drop("Abbreviation", "FullCountryName")

        return df

    def post_process(self, df: DataFrame, parquet_output_path: str, csv_output_path: str):
        df.write.parquet(parquet_output_path, mode="overwrite")
        df.write.csv(csv_output_path, header=True, mode="overwrite")
    
    def run(self):
        df = self.get_table(TURISMO_PATH)
        df = self.treatment_process(df=df)
        self.post_process(df, PARQUET_PATH, CSV_PATH)

if __name__ == '__main__':
    tour = TourismTreatment()
    tour.run()