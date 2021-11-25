# Project 3: Big Data

## Construir los contenedores

```
sudo docker-compose up
```

## Acceder al contenedor master y ubicarse en la carpeta /usr

Acceder al contenedor:
```
sudo docker exec -it <container_id> /bin/bash
```
Ubicarse en la carpeta /usr dentro del contenedor:
```
cd /usr
```

## Data cleaning

Ejectar limpieza de datos
```
spark-2.4.1/bin/spark-submit --master spark://master:7077 /usr/src/data_cleaning.py /tmp/data/Chicago_Crimes_2012_to_2017.csv /tmp/data/cleaned_data/
```
```
spark-2.4.1/bin/spark-submit --master spark://master:7077 /usr/src/data_cleaning.py /tmp/data/Chicago_Crimes_2012_to_2017.csv /tmp/data/cleaned_data_timestamp/
```

## Machine learning
Ejecutar código
```
spark-2.4.1/bin/spark-submit --master spark://master:7077 --jars /usr/src/graphframes-0.8.2-spark2.4-s_2.11.jar /usr/src/machine_learning.py
```