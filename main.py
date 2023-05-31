import string
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import stylecloud
from stop_words import get_stop_words
import altair as alt
from sentence_transformers import SentenceTransformer
from flask import Flask, render_template, request
import mysql.connector

#Los 3 siguientes datos deberian ser los que llegan con el post en app.route (@app.route('/', methods=['POST']))
#Los recojes dentro del app.route asi: nombre = request.form.get('nombre')
alumno = "unai.olaizola@ikasle.aeg.eus" 
nombre= "Unai"
apellido = "Olaizola"

app = Flask(__name__)

@app.route('/')
def index():

    #RECIBIR DATOS SEGUN EL ALUMNO (email)
    conn = mysql.connector.connect(host="c47244.sgvps.net", user='umdcgwdtmlpas', password="L4_3b^3@1q[b", database="dbnfg5oyozhtpn") 
    cursor = conn.cursor()
    consulta = """
               SELECT sentiments.*, ciclo.especialidad
               FROM sentiments
               JOIN alumnosciclos ON sentiments.idalumno = alumnosciclos.idalumno
               JOIN ciclo ON alumnosciclos.idciclo = ciclo.id
               WHERE sentiments.email = %s
               """
    cursor.execute(consulta, (alumno,))
    datos = cursor.fetchall()

    cursor.close()
    conn.close()

    #FUNCION PARA LIMPIAR FRASES
    def clean_string(text):
        if text == "nan":
            return ""
        text = ''.join([word for word in text if word not in string.punctuation])
        text = text.lower()
        return text

    #CARGAR MODELOS
    roberta = f"cardiffnlp/twitter-xlm-roberta-base-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)
    modelS = SentenceTransformer('hiiamsid/sentence_similarity_spanish_es')


    #CLASIFICAR LAS RESPUESTAS
    person_objetives, person_difficulties, person_utilities = [], [], []
    person_all_text = []
    for row in datos:
        person_objetives.append(row[3])
        person_difficulties.append(row[4] + " " + row[5])
        person_utilities.append(row[9] + " " + row[10])
        person_all_text.append(clean_string(row[3]) + ". " + clean_string(row[4]) + ". " + clean_string(row[5]) + ". " + clean_string(row[9]) + ". " + clean_string(row[10]))


    #WORDCLOUDS
    person_wordcloud = " ".join(person_objetives)
    irrelevant_words = get_stop_words("spanish")
    custom_irrelevant_words = irrelevant_words[:]
    custom_irrelevant_words.extend(["hacer","realizar","aprender","aprendido"])

    stylecloud.gen_stylecloud(text=person_wordcloud, custom_stopwords=custom_irrelevant_words, icon_name="fas fa-circle", output_name="static/person.png")
 

    #LEER OBJETIVOS, COMPETENCIAS Y AUTONOMIA
    f1 = open('textos/daw_obj.txt','r', encoding="utf8")
    objetivos_daw = f1.read()
    f2 = open('textos/daw_obj.txt','r', encoding="utf8")
    objetivos_asir = f2.read()
    f3 = open('textos/daw_obj.txt','r', encoding="utf8")
    objetivos_mark = f3.read()
    f4 = open('textos/daw_obj.txt','r', encoding="utf8")
    objetivos_adfin = f4.read()
    f5 = open('textos/daw_obj.txt','r', encoding="utf8")
    objetivos_patro = f5.read()
    f6 = open('textos/daw_obj.txt','r', encoding="utf8")
    objetivos_vestu = f6.read()
    f7 = open('textos/autonomia.txt','r', encoding="utf8")
    autonomia = f7.read()
    f8 = open('textos/participacion.txt','r', encoding="utf8")
    participacion = f8.read()
    f9 = open('textos/compromiso.txt','r', encoding="utf8")
    compromiso = f9.read()


    #OBJETIVOS
    
    if(datos[0][-1]=="DAW"):
        person_objetives.append(objetivos_daw)
    elif(datos[0][-1]=="ASIR"):
        person_objetives.append(objetivos_asir)
    elif(datos[0][-1]=="MARK"):
        person_objetives.append(objetivos_mark)
    elif(datos[0][-1]=="ADFIN"):
        person_objetives.append(objetivos_adfin)
    elif(datos[0][-1]=="VESTU"):
        person_objetives.append(objetivos_vestu)
    elif(datos[0][-1]=="PATRO"):
        person_objetives.append(objetivos_patro)

    
    cleaned = list(map(clean_string, person_objetives))
    embeddings = modelS.encode(cleaned)
    aut = []
    for idx,answer in enumerate(embeddings[:-1]):
        aut.append(cosine_similarity([embeddings[-1]],[embeddings[idx]]))
    obj_similarities = []
    for answer in aut:
        obj_similarities.append(float(answer[0][0]))

    index = [*range(0,len(obj_similarities),1)]

    chart_objetives = pd.DataFrame({
        'x':index,
        'y':obj_similarities
    }
    )
    o = alt.Chart(chart_objetives).mark_area().encode(
        x=alt.X('x', title="Semanas"),
        y=alt.Y('y', title=""),
        color=alt.value("#3399ff")
    ).to_json()

    person_objetives.pop()

    #DIFICULTADES
    difficulties = []
    cleanedD = list(map(clean_string, person_difficulties))
    for idx,answer in enumerate(cleanedD):
        encoded_text = tokenizer(answer, return_tensors='pt')
        output = model(**encoded_text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        difficulties.append(scores)

    color_scale = alt.Scale(
        domain=[
            "positivo",
            "neutral",
            "negativo",
        ],
        range=["#33cc33", "#6699ff", "#ff0000"]
    )

    y_axis = alt.Axis(
        title='Semanas',
        offset=5,
        ticks=False,
        minExtent=60,
        domain=False
    )
    source = []

    for idx,d in enumerate(difficulties):
        start,end = -d[1]/2,d[1]/2
        source.append(
            {
                "question":idx+1,
                "type":"neutral",
                "value":d[1],
                "start":start,
                "end":end
            }
        )
        source.append(
            {
                "question":idx+1,
                "type":"negativo",
                "value":d[0],
                "start":start,
                "end":start-d[0]
            }
        )
        source.append(
            {
                "question":idx+1,
                "type":"positivo",
                "value":d[2],
                "start":end,
                "end":end+d[2]
            }
        )
        

    source = alt.pd.DataFrame(source)


    d = alt.Chart(source).mark_bar().encode(
        x=alt.X('start:Q', title=""),
        x2='end:Q',
        y=alt.Y('question:N', axis=y_axis),
        color=alt.Color(
            'type:N',
            legend=alt.Legend( title='Sentimiento:'),
            scale=color_scale,
        )
    ).to_json()

    #UTILIDAD
    utilities = []
    cleanedU = list(map(clean_string, person_utilities))
    for idx,answer in enumerate(cleanedU):
        encoded_text = tokenizer(answer, return_tensors='pt')
        output = model(**encoded_text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        utilities.append(scores)

    source = []

    for idx,d in enumerate(utilities):
        start,end = -d[1]/2,d[1]/2
        source.append(
            {
                "question":idx+1,
                "type":"neutral",
                "value":d[1],
                "start":start,
                "end":end
            }
        )
        source.append(
            {
                "question":idx+1,
                "type":"negativo",
                "value":d[0],
                "start":start,
                "end":start-d[0]
            }
        )
        source.append(
            {
                "question":idx+1,
                "type":"positivo",
                "value":d[2],
                "start":end,
                "end":end+d[2]
            }
        )
        

    source = alt.pd.DataFrame(source)


    u = alt.Chart(source).mark_bar().encode(
        x=alt.X('start:Q', title=""),
        x2='end:Q',
        y=alt.Y('question:N', axis=y_axis),
        color=alt.Color(
            'type:N',
            legend=alt.Legend( title='Sentimiento:'),
            scale=color_scale,
        )
    ).to_json()

    #AUTONOMIA
    person_all_text.append(autonomia)
    cleanedA = list(map(clean_string, person_all_text))
    embeddings = modelS.encode(cleanedA)
    aut = []
    for idx,answer in enumerate(embeddings[:-1]):
        aut.append(cosine_similarity([embeddings[-1]],[embeddings[idx]]))
    aut_similarities = []
    for answer in aut:
        aut_similarities.append(float(answer[0][0]))

    index = [*range(0,len(aut_similarities),1)]

    chart_autonomia = pd.DataFrame({
        'x':index,
        'y':aut_similarities
    })

    a = alt.Chart(chart_autonomia).mark_area().encode(
        x=alt.X('x', title="Semanas"),
        y=alt.Y('y', title="Nivel de autonomía (0-1)"),
        color=alt.value("#ff6600")
    ).to_json()

    person_all_text.pop()

    #PARTICIPACION

    person_all_text.append(participacion)
    cleanedP = list(map(clean_string, person_all_text))
    embeddings = modelS.encode(cleanedP)
    par = []
    for idx,answer in enumerate(embeddings[:-1]):
        par.append(cosine_similarity([embeddings[-1]],[embeddings[idx]]))
    par_similarities = []
    for answer in par:
        par_similarities.append(float(answer[0][0]))

    chart_participacion =  pd.DataFrame({
        'x':index,
        'y':par_similarities
    })

    p = alt.Chart(chart_participacion).mark_area().encode(
        x=alt.X('x', title="Semanas"),
        y=alt.Y('y', title="Nivel de participación (0-1)"),
        color=alt.value("#33cc33")
    ).to_json()

    person_all_text.pop()

    #COMPROMISO

    person_all_text.append(compromiso)
    cleanedC = list(map(clean_string, person_all_text))
    embeddings = modelS.encode(cleanedC)
    com = []
    for idx,answer in enumerate(embeddings[:-1]):
        com.append(cosine_similarity([embeddings[-1]],[embeddings[idx]]))
    com_similarities = []
    for answer in com:
        com_similarities.append(float(answer[0][0]))

    chart_compromiso = pd.DataFrame({
        'x':index,
        'y':com_similarities
    })

    c = alt.Chart(chart_compromiso).mark_area().encode(
        x=alt.X('x', title="Semanas"),
        y=alt.Y('y', title="Nivel de compromiso (0-1)"),
        color=alt.value("#3399ff")
    ).to_json()

    person_all_text.pop()

    return render_template('test.html', o=o, d=d, u=u, a=a, p=p, c=c, nombre=nombre, apellido=apellido)
    
if __name__ == "__main__":
    app.run(port=8080)