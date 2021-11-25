import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql.functions import *


#definicion funciones
def eliminarColumnas(_data, _Lcolumnas):
    return _data.select([column for column in _data.columns if column not in _Lcolumnas]) 

def eliminarColumnasNulos(_data, _Lcolumnas):
    return _data.dropna(how='any', subset=_Lcolumnas)

def serializarColumnas(_data, _Lcolumnas):
    indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(df) for column in _Lcolumnas]
    pipeline = Pipeline(stages=indexers)
    dfTemp = pipeline.fit(df).transform(df)
    return dfTemp

def deStringaTime(_data, _Lcolumnas):
    for i in _Lcolumnas:
        _data.withColumn(i+" ts",to_timestamp(i))
    return _data

if __name__ == "__main__":
    #inicio sesion spark
    spark=SparkSession.builder.appName('DataCleaning').getOrCreate()

    #carga de datos 
    df = spark.read.csv(sys.argv[1], sep=',', header= True, inferSchema=True)

    #columnas a eliminar
    columnasEliminar = ['_c0', 'ID', 'Case Number', 'Primary Type', 'Description', 'Ward', 'Year', 
                        'Latitude', 'Longitude', 'Location', 'Updated On']
    df = df.select([column for column in df.columns if column not in columnasEliminar])

    #elimanar filas con valores nulos 
    columnasNulos = ['District', 'Community Area', 'X Coordinate', 'Y Coordinate', 'Location Description']
    df = df.dropna(how='any', subset=columnasNulos)

    #serializar columnas
    columnasSerializar = ['IUCR', 'Location Description', 'FBI Code', 'Block']
    indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(df) for column in columnasSerializar]
    pipeline = Pipeline(stages=indexers)
    df = pipeline.fit(df).transform(df)
    df = df.select([column for column in df.columns if column not in columnasSerializar])

    #conervirt booleanos a enteros
    df = df.withColumn("Arrest", when(col("Arrest"), lit(1)).otherwise(0)).withColumn("Domestic", when(col("Domestic"), lit(1) ).otherwise(0))


    #castear str a dateStamp
    columnasDate = ['Date'] 
    df = df.withColumn(columnasDate[0]+" ts",to_timestamp(columnasDate[0], "MM/dd/yyyy hh:mm:ss"))
    df = df.select([column for column in df.columns if column not in columnasDate])

    #extrar componentes de las fechas
    columnasTime = ['Date ts']
    df = df.withColumn("mesDel", month(col("Date ts")))
    df = df.withColumn("diaDel", dayofweek(col("Date ts")))
    df = df.withColumn("horaDel", hour(col("Date ts")))
    df = df.withColumn("minutoDel", minute(col("Date ts")))
    df = df.select([column for column in df.columns if column not in columnasTime])

    #importar datos limpios a csv
    df.write.csv(sys.argv[2], header=True)


#comando para ejecutar script desde servidor (ejecutar desde la carpeta /usr)
#spark-2.4.1/bin/spark-submit --master spark://master:7077 /usr/src/dataClean/dataClean.py /tmp/data/Chicago_Crimes_2012_to_2017.csv /tmp/data/dataClean/