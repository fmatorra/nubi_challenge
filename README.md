# nubi_challenge
Desafio para ingreso a Nubi

## NUBI NLP / ML Engineering Challenge
## ======================================

### Guia de Procedimiento
### ---------------------

* 1. Detectar el tipo de producto:
        Para ello se realizó un clasificador de textos con la libreria transformes y se hizo un Fine-Tunning de un modelo en español. Los datos se extrajeron de ambos CSV (Celular, TVs). Se utilizaron 3 categorias (Televisión, Celular y Otros), debido a que en el dataset se encontraron publicaciones que no correspondian a una de las dos categorias. Modelo, entrenamiento y datos estan en la carpeta ./transformer_classification. Como algo adicional se realizó un clasificador de textos para detectar si el tipo de televisor era o no Smart.
* 2. Detectar entidades nombradas de marcas en el título:
        a) La mejor alternativa para lidiar con los errores de ortografía fue utilizar ner. Para ello se utilizó la libreria spacy v2.3.5, se preentreno con un modelo en español (es_core_news) y se utilizaron ~1200 publicaciones. Modelo, entrenamiento y datos estan en la carpeta ./spacy_ner.
        b) Se raelizó una corrección ortográfica de las NER obtenidas para ello se ralizaron vectores con FastText y se corrigieron según el listado de productos encontrados en ML. Ver archivo ../vectors_ent_correction.ipynb
* 3. Analisis EDA:
        El análisis EDA se dividió en 2 partes, con los datos procesados y sin procesar. Para los datos sin procesar se uso pandas profiling y matplotlib. Para los procesados, se utilizó matplotlib y además se realizó un datastudio para poder hacer una consulta interactiva. Archivos en carpeta ../EDA, datastudio en https://datastudio.google.com/reporting/72957a32-1ecd-45aa-b642-44586bf67d52
* 4. Flask API:
        Se construyo un microservicio con Flask el mismo se divide cuenta dos tipo de consultas.
        a) Consultar http://127.0.0.1:5000/predict?title=**Acá va el título de una publicación** Esto devuelve el la inferencia según los diferentes modelos.
        b) Consultar http://127.0.0.1:5000/report?producto=Televisión devuelve algunos estadísticos del producto o marca consultada.  
