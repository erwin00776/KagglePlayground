__author__ = 'erwin'

import scipy as sp

def log_loss(act, pred):
    epsilon = 1e-10
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    loss = sum(act * sp.log(pred) +
               sp.subtract(1, act) * sp.log(sp.subtract(1, pred)))
    loss = loss * -1.0 / len(act)
    return loss

def abs_loss(act, pred):
    loss = sum(sp.abs(sp.subtract(act, pred)))
    loss = loss * 1.0 / len(act)
    return loss

def square_loss(act, pred):
    loss = sum(sp.subtract(act, pred)**2)
    loss = loss * 1.0 / len(act)
    return loss

def zero_one_loss(act, pred):
    pass

def RMSLE(act, pred):
    loss = sum((sp.log(sp.add(pred, 1)) - sp.log(sp.add(act, 1))) ** 2)
    loss = sp.sqrt(loss * 1.0 / len(pred))
    return loss
