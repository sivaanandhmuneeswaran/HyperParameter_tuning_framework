from django import template
from parameter_tuning import views
import os
from deep_learning import settings
register = template.Library()

@register.simple_tag
def getModel():
    path = os.path.join(settings.BASE_DIR,'media\classify')
    return f'{path}\model.h5'


@register.simple_tag
def getWeight():
    path = os.path.join(settings.BASE_DIR,'media\classify')
    return f'{path}\weights.h5'
