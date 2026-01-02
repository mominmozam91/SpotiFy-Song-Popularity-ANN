from django.shortcuts import render
from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.preprocessing import StandardScaler , LabelEncoder


model = load_model('SpotifAI_ann_project/spotify_ann_model.h5')
ss = StandardScaler()
le = LabelEncoder()

model_columns = ["acousticness", "danceability", "duration_ms", "energy", "instrumentalness", "liveness", "loudness", "speechiness", "tempo", "time_signature", "valence", "genre_Children’s Music", "genre_Electronic", "genre_Folk", "genre_Indie", "genre_Jazz", "genre_Other", "genre_Pop", "genre_Rock", "genre_Soundtrack", "mode_Minor", "key_A#", "key_B", "key_C", "key_C#", "key_D", "key_D#", "key_E", "key_F", "key_F#", "key_G", "key_G#" ]

def index(request):
    if request.method == "POST":
      try:
        danceability = float(request.POST.get("danceability"))
        energy = float(request.POST.get("energy"))
        tempo = float(request.POST.get("tempo"))
        valence = float(request.POST.get("valence"))
        speechiness = float(request.POST.get("speechiness"))
        genre = request.POST.get("genre")
        mode = request.POST.get("mode")
        key = request.POST.get("key")
        time_signature = request.POST.get("time_signature")
        acousticness = 0.30
        duration_ms  = 210000
        instrumentalness  = 0.1
        liveness = 0.15
        loudness = -7.0

        df = pd.DataFrame([[danceability , energy , tempo , valence , speechiness , genre , mode , key , time_signature , acousticness , duration_ms  , instrumentalness , liveness , loudness]] , 
        columns = ['danceability' ,'energy' ,'tempo' ,'valence' ,'speechiness' ,'genre' ,'mode' ,'key' ,'time_signature' , 'acousticness' , 'duration_ms' , 'instrumentalness' , 'liveness' , 'loudness'])
        
        df = pd.get_dummies(df , columns = ['genre' , 'mode' , 'key'] , drop_first = True)

        df['time_signature'] = le.fit_transform(df['time_signature'])

        scaled_cols = ['tempo' , 'duration_ms']
        df[scaled_cols] = ss.fit_transform(df[scaled_cols])
        
        
        for col in model_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[model_columns] 

        prediction = model.predict(df)[0]
        data = {
           'danceability' : danceability,
           'energy' : energy,
           'tempo' : tempo,
           'valence' : valence,
           'speechiness' : speechiness,
           'genre' : genre,
           'mode' : mode,
           'key' : key,
           'time_signature' : time_signature,
           'acousticness' : acousticness,
           'duration_ms' : duration_ms,
           'instrumentalness' : instrumentalness,
           'liveness' : liveness,
           'loudness' : loudness,  
           'prediction' : prediction[0]
        }

        return render(request, 'index.html', data)
    
      except Exception as e:
        return render(request, "index.html", {"error": str(e)})

    return render(request, "index.html")