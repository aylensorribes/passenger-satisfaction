import gradio as gr
import pandas as pd
import pickle

# Define params names
PARAMS_NAME = [
    "Age",
    "Wifi",
    "Booking",
    "Seat",
    "Checkin",
    "Class",
]

# Load model
with open("model/rf_sample_airline.pkl", "rb") as f:
    model = pickle.load(f)

# Columnas
COLUMNS_PATH = "model/categories_ohe_airline.pkl"
with open(COLUMNS_PATH, 'rb') as handle:
    ohe_tr = pickle.load(handle)


def predict(*args):
    answer_dict = {}

    for i in range(len(PARAMS_NAME)):
        answer_dict[PARAMS_NAME[i]] = [args[i]]

    single_instance = pd.DataFrame.from_dict(answer_dict)
    
    # Reformat columns
    single_instance_ohe = pd.get_dummies(single_instance).reindex(columns = ohe_tr).fillna(0)
    
    prediction = model.predict(single_instance_ohe)

    response = format(prediction[0], '.2f')

    return response


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Satisfaccion del pasajero de la aerolinea
        """
    )

    with gr.Row():
        with gr.Column():

            gr.Markdown(
                """
                ## Predecir si el pasajero estará satisfecho o no.
                """
            )
            
            Class = gr.Radio(
                label="Clase del asiento",
                choices=["Business", "Eco", "Eco_Plus"],
                value="H"
                )
            
            Seat = gr.Dropdown(
                label="Comodidad del asiento",
                choices=["0", "1", "2","3","4","5"],
                multiselect=False,
                value="H"
                )
            Age = gr.Slider(label="Edad", minimum=7, maximum=100, step=1, randomize=True)

            Booking = gr.Slider(
                label="Facilidad de la reservacion (0: muy facil, 5:muy dificil)",minimum=0, maximum=5, step=1, randomize=True)     
            
            Wifi = gr.Slider(label="Nivel de satisfaccion con el servicio wifi durante el vuelo (0: insatisfecho, 5:muy satisfecho)",
                minimum=0, maximum=5, step=1, randomize=True)
                

            Checkin = gr.Dropdown(
                label="Nivel de satisfaccion con el proceso de check-in(0: insatisfecho, 5:muy satisfecho)",
                choices=["0", "1", "2","3","4","5"],
                multiselect=False,
                value="H")        
        with gr.Column():

            gr.Markdown(
                """
                ## Predicción
                """
            )

            label = gr.Label(label="Score")
            predict_btn = gr.Button(value="Evaluar")
            predict_btn.click(
            predict,
            inputs=[
                Age,
                Wifi,
                Booking,
                Seat,
                Checkin,
            ],
            outputs=[label],
            )
    gr.Markdown(
        """
        <p style='text-align: center'>
            <a href='https://github.com/aylensorribes' 
                target='_blank'>Proyecto demo creado en el bootcamp de EDVAI 🤗
            </a>
        </p>
        """
    )

demo.launch()
