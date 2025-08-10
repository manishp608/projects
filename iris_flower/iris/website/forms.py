from django import forms

class IrisForm(forms.Form):
    sepal_length = forms.FloatField(label='', widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Sepal Length'}))
    sepal_width = forms.FloatField(label='', widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Sepal Width'}))
    petal_length = forms.FloatField(label='', widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Petal Length'}))
    petal_width = forms.FloatField(label='', widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Petal Width'}))