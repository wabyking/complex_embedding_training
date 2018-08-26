# -*- coding: utf-8 -*-

import traceback,sys,os




def setup(opt,helper):
    config = {k:v   for conf in (opt,helper) for k,v in conf.__dict__.items()}
    model = import_object(opt.network_type, config)        
    return model

def import_class(import_str):
    dirname, filename = os.path.split(os.path.abspath(__file__))
    sys.path.insert(0,dirname)
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' %
                (class_str,
                    traceback.format_exception(*sys.exc_info())))


def import_object(import_str, *args, **kwargs):
    return import_class(import_str)(*args, **kwargs)



if __name__ == "__main__":
    from params import Params
    params = Params()
    config_file = 'config/waby.ini'    # define dataset in the config
    params.parse_config(config_file)


    
    from dataset import qa
    reader = qa.setup(params)
    from models.match import keras as models
    model = models.setup(params)