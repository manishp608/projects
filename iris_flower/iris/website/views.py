from django.shortcuts import render
from .forms import IrisForm
import joblib
import os
import numpy as np
from django.conf import settings

model_path = os.path.join(settings.BASE_DIR, 'website', 'models', 'svm_knn_model.pkl')
scalar_path = os.path.join(settings.BASE_DIR, 'website', 'models', 'scalar.pkl')
knn_model = joblib.load(model_path)
scalar = joblib.load(scalar_path)

# Create your views here.
def home(request):
    return render(request, 'home.html', {})

def predict_iris(request):
    if request.method == 'POST':
        form = IrisForm(request.POST)
        if form.is_valid():
            data = np.array([
                form.cleaned_data['sepal_length'],
                form.cleaned_data['sepal_width'],
                form.cleaned_data['petal_length'],
                form.cleaned_data['petal_width']
            ]).reshape(1, -1)

            data = scalar.transform(data)

            prediction = knn_model.predict(data)[0]
            print(prediction)
            species = ['Setosa', 'Versicolor', 'Virginica'][prediction]

            return render(request, 'result.html', {'species': species})
    else:
        form = IrisForm()

    return render(request, 'predict.html', {'form': form})



