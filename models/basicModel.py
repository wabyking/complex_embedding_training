# -*- coding: utf-8 -*-

class BasicModel(object):
    
    def __init__(self,config):
        self.config= {}
        for k,v in config.items():
            self.set_default(k,v)
        
    def set_default(self, k, v):
        if k not in self.config:
            self.config[k] = v
    
    def build(self):
        pass