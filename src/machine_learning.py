import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import *
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from pyspark import SparkConf, SparkContext
from pyspark.sql import Row
from pyspark.sql.functions import monotonically_increasing_id
from graphframes import *

def create_edges(vertices):
    connections = list()
    vertices_collection = vertices.collect()
    for v1 in vertices_collection:
        for v2 in vertices_collection:
            if(v1 != v2):
                if(v1["CommunityArea"] == v2["CommunityArea"] and v1["District"] == v2["District"]):
                    connections.append((v1["id"], v2["id"], "CommunityArea"))
    edges = spark.createDataFrame(connections, ["src", "dst", "relationship"])
    return edges

def parseVertices(line):
    field = line.split(",")
    return Row(Arrest = int(field[0]), Domestic = int(field[1]), Beat = int(field[2]), District = float(field[3]), CommunityArea = float(field[4]), XCoordinate = float(field[5]), YCoordinate = float(field[6]), IUCR_index = float(field[7]), LocationDescription_index = float(field[8]), FBICode_index = float(field[9]), Block_index = float(field[10]), mesDel = int(field[11]), diaDel = int(field[12]), horaDel = int(field[13]), minutoDel = int(field[14]))

if __name__ == "__main__":
    conf = SparkConf().setAppName("MachineLearning")
    sc = SparkContext(conf = conf)
    #inicio sesion spark
    spark=SparkSession.builder.appName('MachineLearning').getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    #carga de datos
    # df = spark.read.csv("/user/maria_dev/ml-graph/data/cleaned_data/*.csv", sep=',', header= True, inferSchema=True)
    lines = spark.sparkContext.textFile("/tmp/data/cleaned_data/*.csv")
    header = lines.first()
    lines = lines.filter(lambda line: line != header)

    vertices_rdd = lines.map(parseVertices)
    vertices = spark.createDataFrame(vertices_rdd).withColumn("id", monotonically_increasing_id())
    vertices.show()

    edges = create_edges(vertices)
    # edges = spark.createDataFrame([(1, 2, "friends"), (1, 3, "friends")], ["src", "dst", "relationship"])

    g = GraphFrame(vertices, edges)

    print(g.edges.count())

    # #vectorizacion de los atributos
    # vector = VectorAssembler(inputCols = ['Domestic', 'Beat', 'District', 'Community Area', 'X Coordinate', 'Y Coordinate', 
    #                                       'IUCR_index', 'Location Description_index', 'FBI Code_index', 'Block_index', 
    #                                       'mesDel', 'diaDel', 'horaDel', 'minutoDel'], outputCol = 'atributos')
    # df = vector.transform(df)
    # df = df.select('atributos', 'Arrest')
    # df = df.selectExpr("atributos as atributos", "Arrest as label")

    # #division del dataset 70% entrenamiento - 30% pruebas
    # train, test = df.randomSplit([0.7, 0.3], seed = 2018)

    # #instacia del evaluador
    # evaluator = BinaryClassificationEvaluator()

    # #regresion logistica
    # lr = LogisticRegression(featuresCol = 'atributos', labelCol = 'label', maxIter=10)
    # lrModel = lr.fit(train)
    # predictions = lrModel.transform(test)
    # accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(predictions.count())
    # print("=========================================================================================================================")
    # print("REGRESION LOGISTICA")
    # print('area bajo el ROC', evaluator.evaluate(predictions))
    # print("presicion de la regresion logistica: ", accuracy)
    # print("=========================================================================================================================")


    # #arboles de decision
    # dt = DecisionTreeClassifier(featuresCol = 'atributos', labelCol = 'label', maxDepth = 3)
    # dtModel = dt.fit(train)
    # predictionsDt = dtModel.transform(test)
    # accuracy2 = predictionsDt.filter(predictionsDt.label == predictionsDt.prediction).count() / float(predictionsDt.count())
    # print("=========================================================================================================================")
    # print("ARBOLES DE DECISION")
    # print("Test Area Under ROC: " + str(evaluator.evaluate(predictionsDt, {evaluator.metricName: "areaUnderROC"})))
    # print("presicion de los arboles de decision: ", accuracy2)
    # print("=========================================================================================================================")

    # #random forest 
    # rf = RandomForestClassifier(featuresCol = 'atributos', labelCol = 'label')
    # rfModel = rf.fit(train)
    # predictionsRf = rfModel.transform(test)
    # accuracy3 = predictionsRf.filter(predictionsRf.label == predictionsRf.prediction).count() / float(predictionsRf.count())
    # print("=========================================================================================================================")
    # print("RANDOM FOREST") 
    # print("Test Area Under ROC: " + str(evaluator.evaluate(predictionsRf, {evaluator.metricName: "areaUnderROC"}))) 
    # print("presicion random forest: ", accuracy3)
    # print("=========================================================================================================================")