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
from pyspark.sql import SQLContext

import time

def create_edges_naive(vertices):
    connections = list()
    vertices_collection = vertices.collect()
    for v1 in vertices_collection:
        for v2 in vertices_collection:
            if(v1 != v2):
                if(v1["CommunityArea"] == v2["CommunityArea"] and v1["District"] == v2["District"]):
                    connections.append((v1["id"], v2["id"], "CommunityArea"))
    edges = spark.createDataFrame(connections, ["src", "dst", "relationship"])
    return edges

def create_edges(vertices, sqlContext):
    vertices.createOrReplaceTempView("vertices")
    # vertices1.mesDel = vertices2.mesDel AND vertices1.FBICode_index = vertices2.FBICode_index AND vertices1.Block_index = vertices2.Block_index AND vertices1.LocationDescription_index = vertices2.LocationDescription_index
    edges = sqlContext.sql("SELECT vertices1.id AS src, vertices2.id AS dst FROM vertices AS vertices1,vertices AS vertices2 WHERE vertices1.CommunityArea = vertices2.CommunityArea AND vertices1.District = vertices2.District AND vertices1.Beat = vertices2.Beat AND vertices1.mesDel = vertices2.mesDel")
    return edges

def parseVertices(line):
    field = line.split(",")
    return Row(Date = field[0], Arrest = int(field[1]), Domestic = int(field[2]), Beat = int(field[3]), District = float(field[4]), CommunityArea = float(field[5]), XCoordinate = float(field[6]), YCoordinate = float(field[7]), IUCR_index = float(field[8]), LocationDescription_index = float(field[9]), FBICode_index = float(field[10]), Block_index = float(field[11]), mesDel = int(field[12]), diaDel = int(field[13]), horaDel = int(field[14]), minutoDel = int(field[15]))

if __name__ == "__main__":
    conf = SparkConf().setAppName("MachineLearning")
    sc = SparkContext(conf = conf)
    sqlContext = SQLContext(sc)
    #inicio sesion spark
    spark=SparkSession.builder.appName('MachineLearning').getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    #carga de datos
    # df = spark.read.csv("/user/maria_dev/ml-graph/data/cleaned_data/*.csv", sep=',', header= True, inferSchema=True)
    lines = spark.sparkContext.textFile("/tmp/data/cleaned_data_timestamp/*.csv")
    header = lines.first()
    lines = lines.filter(lambda line: line != header)

    vertices_rdd = lines.map(parseVertices)
    vertices = spark.createDataFrame(vertices_rdd).withColumn("id", monotonically_increasing_id())
    vertices = vertices.withColumn("timestamp", to_timestamp("Date", "MM/dd/yyyy hh:mm:ss"))
    vertices = spark.createDataFrame(vertices.orderBy("timestamp").take(500))
    vertices.createOrReplaceTempView("temp_vertices")
    sqlContext.sql("SELECT MAX(timestamp) FROM temp_vertices").show()
    vertices = vertices.select([column for column in vertices.columns if column not in {"Date", "timestamp"}])
    df = spark.createDataFrame(vertices.collect())
    vertices.show()
    # df.show()

    print("------------ CREATING EDGES ------------")
    start_time = time.time()
    edges = create_edges(vertices, sqlContext)
    # edges = spark.createDataFrame([(1, 2, "friends"), (1, 3, "friends")], ["src", "dst", "relationship"])
    print("elapsed: {0} seconds".format(time.time() - start_time))
    print("------------ FINIHED CREATING EDGES ------------")

    print("------------ CREATING GRAPH ------------")
    start_time = time.time()
    g = GraphFrame(vertices, edges)
    print("Number of edges: ", g.edges.count())
    print("elapsed: {0} seconds".format(time.time() - start_time))    
    print("------------ FINIHSED CREATING GRAPH ------------")

    print("------------ CALCULATING INDEGREE ------------")
    start_time = time.time()
    indegrees = g.inDegrees
    indegrees.show()
    print("elapsed: {0} seconds".format(time.time() - start_time))
    print("------------ FINISHED CALCULATING INDEGREE ------------")

    print("------------ CALCULATING OUTDEGREE ------------")
    start_time = time.time()
    outdegrees = g.outDegrees
    outdegrees.show()
    print("elapsed: {0} seconds".format(time.time() - start_time))
    print("------------ FINISHED CALCULATING OUTDEGREE ------------")

    print("------------ CALCULATING DEGREE ------------")
    start_time = time.time()
    degrees = g.degrees
    degrees.show()
    print("elapsed: {0} seconds".format(time.time() - start_time))
    print("------------ FINISHED CALCULATING DEGREE ------------")

    print("------------ CALCULATING STRONGLY CONNECTED COMPONENTS ------------")
    start_time = time.time()
    stronglyConnectedComponets = g.stronglyConnectedComponents(maxIter=10)
    stronglyConnectedComponets = stronglyConnectedComponets.select([column for column in stronglyConnectedComponets.columns if column not in {'Arrest', 'Beat','Block_index', 'CommunityArea', 'District', 'Domestic', 'FBICode_index', 'IUCR_index', 'LocationDescription_index', 'XCoordinate', 'YCoordinate', 'diaDel', 'horaDel', 'mesDel', 'minutoDel'}])
    stronglyConnectedComponets.show()
    print("elapsed: {0} seconds".format(time.time() - start_time))
    print("------------ FINISHED CALCULATING STRONGLY CONNECTED COMPONENTS ------------")

    print("------------ CALCULATING PAGERANK ------------")
    start_time = time.time()
    pagerank = g.pageRank(resetProbability=0.15, maxIter=10).vertices
    pagerank = pagerank.select([column for column in pagerank.columns if column not in {'Arrest', 'Beat','Block_index', 'CommunityArea', 'District', 'Domestic', 'FBICode_index', 'IUCR_index', 'LocationDescription_index', 'XCoordinate', 'YCoordinate', 'diaDel', 'horaDel', 'mesDel', 'minutoDel'}])
    pagerank.show()
    print("elapsed: {0} seconds".format(time.time() - start_time))
    print("------------ FINISHED CALCULATING PAGERANK ------------")

    print("------------ CALCULATING LABEL PROPAGATION (COMMUNITIES) ------------")
    start_time = time.time()
    communities = g.labelPropagation(maxIter=5)
    communities = communities.select([column for column in communities.columns if column not in {'Arrest', 'Beat','Block_index', 'CommunityArea', 'District', 'Domestic', 'FBICode_index', 'IUCR_index', 'LocationDescription_index', 'XCoordinate', 'YCoordinate', 'diaDel', 'horaDel', 'mesDel', 'minutoDel'}])
    communities.createOrReplaceTempView("communities")
    communities = sqlContext.sql("SELECT id, label AS community FROM communities")
    communities.show()
    print("elapsed: {0} seconds".format(time.time() - start_time))
    print("------------ FINISHED CALCULATING LABEL PROPAGATION (COMMUNITIES) ------------")

    df.createOrReplaceTempView("df")
    indegrees.createOrReplaceTempView("indegrees")
    outdegrees.createOrReplaceTempView("outdegrees")
    stronglyConnectedComponets.createOrReplaceTempView("stronglyConnectedComponets")
    pagerank.createOrReplaceTempView("pagerank")
    communities.createOrReplaceTempView("communities")
    df = sqlContext.sql("SELECT * FROM df NATURAL JOIN indegrees")
    df.createOrReplaceTempView("df")
    df = sqlContext.sql("SELECT * FROM df NATURAL JOIN outdegrees")
    df.createOrReplaceTempView("df")
    df = sqlContext.sql("SELECT * FROM df NATURAL JOIN stronglyConnectedComponets")
    df.createOrReplaceTempView("df")
    df = sqlContext.sql("SELECT * FROM df NATURAL JOIN pagerank")
    df.createOrReplaceTempView("df")
    df = sqlContext.sql("SELECT * FROM df NATURAL JOIN communities")
    df.show()

    # print("------------ CALCULATING CONNECTED COMPONENTS ------------")
    # sc.setCheckpointDir("/tmp/graphframes-connected-components")
    # start_time = time.time()
    # connectedComponets = g.connectedComponents()
    # connectedComponets.show()
    # print("elapsed: {0} seconds".format(time.time() - start_time))
    # print("------------ FINISHED CALCULATING CONNECTED COMPONENTS ------------")

    # print("------------ CALCULATING TRIANGLE COUNT ------------")
    # start_time = time.time()
    # triangleCount = g.triangleCount()
    # triangleCount.show()
    # print("elapsed: {0} seconds".format(time.time() - start_time))
    # print("------------ FINISHED CALCULATING TRIANGLE COUNT ------------")

    df = df.select([column for column in df.columns if column not in {"id"}])

    #vectorizacion de los atributos
    vector = VectorAssembler(inputCols = ['Arrest', 'Beat','Block_index', 'CommunityArea', 'District', 'Domestic', 'FBICode_index', 'IUCR_index', 'LocationDescription_index', 'XCoordinate', 'YCoordinate', 'diaDel', 'horaDel', 'mesDel', 'minutoDel', 'inDegree', 'outDegree', 'component', 'pagerank', 'community'], outputCol = 'atributos') # 'inDegree', 'outDegree', 'component', 'pagerank', 'community'
    df = vector.transform(df)
    df = df.select('atributos', 'Arrest')
    df = df.selectExpr("atributos as atributos", "Arrest as label")

    #division del dataset 70% entrenamiento - 30% pruebas
    train, test = df.randomSplit([0.7, 0.3], seed = 2018)

    #instacia del evaluador
    evaluator = BinaryClassificationEvaluator()

    #regresion logistica
    lr = LogisticRegression(featuresCol = 'atributos', labelCol = 'label', maxIter=10)
    lrModel = lr.fit(train)
    predictions = lrModel.transform(test)
    accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(predictions.count())
    print("=========================================================================================================================")
    print("REGRESION LOGISTICA")
    print('area bajo el ROC', evaluator.evaluate(predictions))
    print("presicion de la regresion logistica: ", accuracy)
    print("=========================================================================================================================")


    #arboles de decision
    dt = DecisionTreeClassifier(featuresCol = 'atributos', labelCol = 'label', maxDepth = 3)
    dtModel = dt.fit(train)
    predictionsDt = dtModel.transform(test)
    accuracy2 = predictionsDt.filter(predictionsDt.label == predictionsDt.prediction).count() / float(predictionsDt.count())
    print("=========================================================================================================================")
    print("ARBOLES DE DECISION")
    print("Test Area Under ROC: " + str(evaluator.evaluate(predictionsDt, {evaluator.metricName: "areaUnderROC"})))
    print("presicion de los arboles de decision: ", accuracy2)
    print("=========================================================================================================================")

    #random forest 
    rf = RandomForestClassifier(featuresCol = 'atributos', labelCol = 'label')
    rfModel = rf.fit(train)
    predictionsRf = rfModel.transform(test)
    accuracy3 = predictionsRf.filter(predictionsRf.label == predictionsRf.prediction).count() / float(predictionsRf.count())
    print("=========================================================================================================================")
    print("RANDOM FOREST") 
    print("Test Area Under ROC: " + str(evaluator.evaluate(predictionsRf, {evaluator.metricName: "areaUnderROC"}))) 
    print("presicion random forest: ", accuracy3)
    print("=========================================================================================================================")