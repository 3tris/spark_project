from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def clean_data(df: DataFrame) -> DataFrame:
    """Clean the input DataFrame.

    - Removes rows where amount <= 0 (rows with NULL amount are removed too,
      since NULL > 0 evaluates to NULL and the filter drops it).
    - Removes rows where name is NULL.
    - Adds amount_with_tax = amount * 1.20.
    """
    return (
        df.filter(F.col("amount") > 0)
        .filter(F.col("name").isNotNull())
        .withColumn("amount_with_tax", F.col("amount") * 1.20)
    )
