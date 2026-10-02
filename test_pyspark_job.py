import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

from pyspark_job import clean_data

SCHEMA = StructType(
    [
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("amount", DoubleType(), True),
    ]
)


@pytest.fixture(scope="session")
def spark():
    session = (
        SparkSession.builder.master("local[1]")
        .appName("pyspark-ci-tests")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()


def test_valid_records_are_kept(spark):
    df = spark.createDataFrame([(1, "Alice", 100.0), (2, "Bob", 50.0)], SCHEMA)
    result = clean_data(df)
    assert result.count() == 2
    assert {r["name"] for r in result.collect()} == {"Alice", "Bob"}


def test_amount_less_than_or_equal_zero_removed(spark):
    df = spark.createDataFrame(
        [(1, "Alice", 100.0), (2, "Bob", 0.0), (3, "Carol", -10.0)], SCHEMA
    )
    result = clean_data(df)
    rows = result.collect()
    assert len(rows) == 1
    assert rows[0]["name"] == "Alice"


def test_null_names_removed(spark):
    df = spark.createDataFrame([(1, "Alice", 100.0), (2, None, 200.0)], SCHEMA)
    result = clean_data(df)
    rows = result.collect()
    assert len(rows) == 1
    assert rows[0]["name"] == "Alice"


def test_amount_with_tax_calculated_correctly(spark):
    df = spark.createDataFrame([(1, "Alice", 100.0), (2, "Bob", 50.0)], SCHEMA)
    result = clean_data(df)
    assert "amount_with_tax" in result.columns
    values = {r["name"]: r["amount_with_tax"] for r in result.collect()}
    assert values["Alice"] == pytest.approx(120.0)
    assert values["Bob"] == pytest.approx(60.0)
