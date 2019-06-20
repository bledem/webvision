# -*- coding: utf-8 -*-
import torchvision.models as models
import torch.nn as nn

#Finally, notice that inception_v3 requires the input size to be (299,299),
# whereas all of the other models expect (224,224).



"""Factory method for easily getting models by name."""

__all__ = ['densenet121', 'densenet169', 'densenet201', 'densenet161',
           'inception_v3', 'resnet18', 'resnet34', 'resnet50',
           'resnet101', 'resnet152', 'vgg11', 'vgg11_bn', 'vgg13',
           'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19', 'alexnet']  #as simply alexnet as in https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html


def get_model(name, num_classes, **kwargs):
    """Get a model by name."""
    if name not in __all__:
        raise NotImplementedError
    params = ""
    for key in kwargs.keys():
        params += (key + '=' + str(kwargs[key]) + ', ')
    
    if name.startswith('dense'):
        exec("from densenet import {}".format(name))
    elif name.startswith('incep'):
        model = models.inception_v3(params)
        model.AuxLogits.fc = nn.Linear(768, num_classes)
        model.fc = nn.Linear(2048, num_classes)
    elif name.startswith('resnet'):
        print(params)
        model = models.resnet50( pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif name.startswith('vgg'):
        model = models.vgg11_bn(params)
        model.classifier[6] = nn.Linear(4096,num_classes)

    elif name.startswith('alexnet'):
        model = models.alexnet(params + "pretrained = False")
        model.classifier[6] = nn.Linear(4096,num_classes)


    else:
        raise NotImplementedError

   

   # model = eval(name + '(' + params[:-2] + ')')

    return model
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    print("Loading a model..")

    if model_name == "resnet":
        """ Resnet18
        """

        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        print("in resnet50")
        print(model_ft)
        print(num_ftrs)
        model_ft.fc = nn.Linear(num_ftrs, 2203)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,2203)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size



def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            #By default, when we load a pretrained model all of the parameters 
            #have .requires_grad=True, which is fine if we are training from scratch or finetuning
            param.requires_grad = False 

