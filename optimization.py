#includes battery model that doesn't work 


from gurobipy import *


# In[443]:


model = Model()
model.params.NonConvex = 2


# In[444]:


k = 1
T = 20
V_max = 1.1
f_max = 10.0
c_batt = 1 / (V_max**2 * f_max) 
epsilon = 0.0001
FEP = 2.4


# In[445]:


f = model.addVar(lb = 0.01, ub=f_max, name = "f")
x = model.addVar(lb = epsilon, ub=5000,name = "x")


# In[446]:


# Battery regression model parameter dump
aI2, aI1, aI0 = -5.851e-5,-1.513e-4,-2.535e-4
bI2, bI1, bI0 = -1.442e-3,-5.299e-2,2.918
cI2, cI1, cI0 = -3.589e-7,9.463e-7,-4.831e-7
dI2, dI1, dI0 = 1.077e-2,-3.443e-2,6.645e-2
eI2, eI1, eI0 = -2.470,6.330,7.267

aT3, aT2, aT1, aT0 = -3.83e-7,2.79e-5,-5.54e-4,1.31e-3
bT3, bT2, bT1, bT0 = 2.19e-7,8.24e-6,-3.67e-4,2.75
cT3, cT2, cT1, cT0 = 4.13e-9,-3.36e-7,8.02e-6,-7.25e-5
dT3, dT2, dT1, dT0 = 9.36e-6,-6.65e-4,1.28e-2,-1.17e-2
eT3, eT2, eT1, eT0 = -6.24e-4,4.50e-2,-8.69e-1,8.64

aDC3, aDC2, aDC1, aDC0 = 3.620e-1,-3.972e-1,9.542e-2,-6.841e-3
bDC3, bDC2, bDC1, bDC0 = -3.034e-2,1.391,-1.524,2.905
cDC3, cDC2, cDC1, cDC0 = -4.764e-6,4.048e-6,-8.439e-7,1.896e-9
dDC3, dDC2, dDC1, dDC0 = -12.11,13.53,-3.163,0.2374
eDC3, eDC2, eDC1, eDC0 = -8.449e2,7.033e2,-1.656e2,2.300e1


a_vs_I = {
    0.5 : -0.000309,
    1.0 : -0.000481,
    1.5 : -0.000692,
    2.0 : -0.000796,
    2.5 : -0.000851,
    3.0 : -0.001312
}

b_vs_I = {
    0.5 : 2.8938,
    1.0 : 2.8594,
    1.5 : 2.8346, 
    2.0 : 2.8078,
    2.5 : 2.7798, 
    3.0 : 2.7438
}

c_vs_I = {
    0.5 : -0.205116e-9, 
    1.0 : -0.492248e-9,
    1.5 : -0.640831e-9,
    2.0 : -4.392784e-9,
    2.5 : -94.584467e-9, 
    3.0 : -1027.852434e-9, 
}

d_vs_I = {
    0.5 : 0.0470, 
    1.0 : 0.0492, 
    1.5 : 0.0436, 
    2.0 : 0.0373, 
    2.5 : 0.0394, 
    3.0 : 0.0656 
}

e_vs_I = {
    0.5 : 10.7261,
    1.0 : 9.6671,
    1.5 : 10.8093,
    2.0 : 10.8657,
    2.5 : 8.4749,
    3.0 : 3.3261
}

a_vs_T = {
    5 : -2.901868e-4, 
    10 : -29.85632e-4,
    15 : -18.67720e-4,
    20 : -8.303639e-4,
    25 : -11.71509e-4,
    30 : -7.247482e-4,
    35 : -8.857543e-4,
    40 : -5.037751e-4
}

b_vs_T = {
    5 : 2.744198, 
    10 : 2.759607, 
    15 : 2.746171, 
    20 : 2.738122, 
    25 : 2.754787, 
    30 : 2.752003, 
    35 : 2.762291, 
    40 : 2.760638 
}

c_vs_T = {
    5 : -4.896851e-5, 
    10 : -0.02172541e-5,
    15 : -2.205842e-5,
    20 : -2.459526e-5,
    25 : -1.639553e-5,
    30 : -1.734742e-5,
    35 : -2.356738e-5,
    40 : -2.822796e-5
}

d_vs_T = {
    5 : 2.345467e-2,
    10 : 9.115434e-2, 
    15 : 5.330070e-2, 
    20 : 3.560531e-2, 
    25 : 3.849303e-2, 
    30 : 3.487165e-2, 
    35 : 3.145855e-2, 
    40 : 2.983178e-2 
}

e_vs_T = {
    5 : 5.816614,
    10 : 2.939495,
    15 : 3.175000,
    20 : 5.491996,
    25 : 5.293104,
    30 : 5.934819,
    35 : 5.928847,
    40 : 6.210430
}

a_vs_DC = {
    0.1 :-4.361987e-4, 
    0.2 : -26.35938e-4,
    0.3 : -13.52700e-4,
    0.4 : -109.3986e-4,
    0.5 : -127.0010e-4
}

b_vs_DC = {
    0.1 : 2.766759,
    0.2 : 2.653940,
    0.3 : 2.574400,
    0.4 : 2.514378,
    0.5 : 2.487273
}

c_vs_DC = {
    0.1 : -44.94255e-9, 
    0.2 : -50.18709e-9, 
    0.3 : -4.928452e-9, 
    0.4 : -9.459073e-12,
    0.5 : -1.810338e-9
}

d_vs_DC = {
    0.1 : 3.376468e-2, 
    0.2 : 9.135696e-2, 
    0.3 : 11.60005e-2, 
    0.4 : 40.42848e-2, 
    0.5 : 51.43958e-2 
}

e_vs_DC = {
    0.1 : 1.271104e1,
    0.2 : 1.090703e1,
    0.3 : 1.431877e1,
    0.4 : 1.503630e1,
    0.5 : 1.049961e1
}


def gen_cvals():
    cvals = {}

    #generate scaling constants for a
    sumCI = 0
    for (I,a_exp) in a_vs_I.items():
        a_reg = aI2 * I**2 + aI1 * I + aI0
        sumCI += a_reg / a_exp
    CI_a = sumCI / len(a_vs_I)

    sumCT = 0
    for (T,a_exp) in a_vs_T.items():
        a_reg = aT3 * T**3 + aT2 * T**2 + aT1 * T + aT0
        sumCT += a_reg / a_exp
    CT_a = sumCT / len(a_vs_T)

    sumCDC = 0
    for (DC,a_exp) in a_vs_DC.items():
        a_reg = aDC3 * DC * DC * DC + aDC2 * DC * DC + aDC1 * DC + aDC0
        sumCDC += a_reg / a_exp
    CDC_a = sumCDC / len(a_vs_DC)

    cvals['a'] = [CI_a,CT_a,CDC_a]

    #generate scaling constants for b
    sumCI = 0
    for (I,b_exp) in b_vs_I.items():
        b_reg = bI2 * I**2 + bI1 * I + bI0
        sumCI += b_reg / b_exp
    CI_b = sumCI / len(b_vs_I)

    sumCT = 0
    for (T,b_exp) in b_vs_T.items():
        b_reg = bT3 * T**3 + bT2 * T**2 + bT1 * T + bT0
        sumCT += b_reg / b_exp
    CT_b = sumCT / len(b_vs_T)

    sumCDC = 0
    for (DC,b_exp) in b_vs_DC.items():
        b_reg = bDC3 * DC * DC * DC + bDC2 * DC * DC + bDC1 * DC + bDC0
        sumCDC += b_reg / b_exp
    CDC_b = sumCDC / len(b_vs_DC)

    cvals['b'] = [CI_b,CT_b,CDC_b]

    #generate scaling constants for c
    sumCI = 0
    for (I,c_exp) in c_vs_I.items():
        c_reg = cI2 * I**2 + cI1 * I + cI0
        sumCI += c_reg / c_exp
    CI_c = sumCI / len(c_vs_I)

    sumCT = 0
    for (T,c_exp) in c_vs_T.items():
        c_reg = cT3 * T**3 + cT2 * T**2 + cT1 * T + cT0
        sumCT += c_reg / c_exp
    CT_c = sumCT / len(c_vs_T)

    sumCDC = 0
    for (DC,c_exp) in c_vs_DC.items():
        c_reg = cDC3 * DC * DC * DC + cDC2 * DC * DC + cDC1 * DC + cDC0
        sumCDC += c_reg / c_exp
    CDC_c = sumCDC / len(c_vs_DC)

    cvals['c'] = [CI_c,CT_c,CDC_c]

    #generate scaling constants for d
    sumCI = 0
    for (I,d_exp) in d_vs_I.items():
        d_reg = dI2 * I**2 + dI1 * I + dI0
        sumCI += d_reg / d_exp
    CI_d = sumCI / len(d_vs_I)

    sumCT = 0
    for (T,d_exp) in d_vs_T.items():
        d_reg = dT3 * T**3 + dT2 * T**2 + dT1 * T + dT0
        sumCT += d_reg / d_exp
    CT_d = sumCT / len(d_vs_T)

    sumCDC = 0
    for (DC,d_exp) in d_vs_DC.items():
        d_reg = dDC3 * DC * DC * DC + dDC2 * DC * DC + dDC1 * DC + dDC0
        sumCDC += d_reg / d_exp
    CDC_d = sumCDC / len(d_vs_DC)

    cvals['d'] = [CI_d,CT_d,CDC_d]

    #generate scaling constants for e
    sumCI = 0
    for (I,e_exp) in e_vs_I.items():
        e_reg = eI2 * I**2 + eI1 * I + eI0
        sumCI += e_reg / e_exp
    CI_e = sumCI / len(e_vs_I)

    sumCT = 0
    for (T,e_exp) in e_vs_T.items():
        e_reg = eT3 * T**3 + eT2 * T**2 + eT1 * T + eT0
        sumCT += e_reg / e_exp
    CT_e = sumCT / len(e_vs_T)

    sumCDC = 0
    for (DC,e_exp) in e_vs_DC.items():
        e_reg = eDC3 * DC * DC * DC + eDC2 * DC * DC + eDC1 * DC + eDC0
        sumCDC += e_reg / e_exp
    CDC_e = sumCDC / len(e_vs_DC)

    cvals['e'] = [CI_e,CT_e,CDC_e]

    return cvals

cvals = gen_cvals()


# In[447]:


# Create a new variable for SOC voltage
V_soc = 1.1
#model.addVar(lb=0.5, ub=1.1,name="V_soc")
#model.addConstr(V_soc == k * f, "soc_voltage_def")

# # Define V_sq = V^2
# V_soc_sq = model.addVar(name="V_sq")
# model.addGenConstrPow(V_soc, V_soc_sq, 2.0, name="V_soc_squared")

# # Define V_sq_f = V^2 * f
# V_soc_sq_f = model.addVar(name="V_soc_sq_f")
# model.addConstr(V_soc_sq_f == V_soc_sq * f, name="V_soc_sq_f_def")

# Define the rest of the parameters
#100uW - 10mW
P = model.addVar(lb=0.1, ub=3.3, name="P")
model.addConstr(P == 0.1 + 0.9 * c_batt * V_soc * V_soc*f)

I = model.addVar(lb=0.5, ub=3.0, name="I")
model.addConstr(I == P / V_soc, name="current_def")

DC = 0.3
#model.addVar(lb=0.1, ub=0.5, name="DC")
#model.addConstr(DC * f * T >= 0.5)
#model.addConstr(DC * f * T <= 2.0)


# In[448]:


#Encode battery model parameters

a_I_expr = aI2 * I * I + aI1 * I + aI0
a_DC_expr = aDC3 * DC * DC * DC + aDC2 * DC * DC + aDC1 * DC + aDC0
a_T_expr = aT3 * T**3 + aT2 * T**2 + aT1 * T + aT0
a = model.addVar(name="a")
CI_a, CT_a, CDC_a = cvals['a']
model.addConstr(a == (a_I_expr * a_DC_expr * a_T_expr) / (CI_a * CT_a * CDC_a), name="a_def")

b_I_expr = bI2 * I * I + bI1 * I + bI0
b_DC_expr = bDC3 * DC * DC * DC + bDC2 * DC * DC + bDC1 * DC + bDC0
b_T_expr = bT3 * T**3 + bT2 * T**2 + bT1 * T + bT0
b = model.addVar(name="b")
CI_b, CT_b, CDC_b = cvals['b']
model.addConstr(b == (b_I_expr * b_DC_expr * b_T_expr) / (CI_b * CT_b * CDC_b), name="b_def")

c_I_expr = cI2 * I * I + cI1 * I + cI0
c_DC_expr = cDC3 * DC * DC * DC + cDC2 * DC * DC + cDC1 * DC + cDC0
c_T_expr = cT3 * T**3 + cT2 * T**2 + cT1 * T + cT0
c = model.addVar(name="c")
CI_c, CT_c, CDC_c = cvals['c']
model.addConstr(c == (c_I_expr * c_DC_expr * c_T_expr) / (CI_c * CT_c * CDC_c), name="c_def")

d_I_expr = dI2 * I * I + dI1 * I + dI0
d_DC_expr = dDC3 * DC * DC * DC + dDC2 * DC * DC + dDC1 * DC + dDC0
d_T_expr = dT3 * T**3 + dT2 * T**2 + dT1 * T + dT0
d = model.addVar(name="d")
CI_d, CT_d, CDC_d = cvals['d']
model.addConstr(d == (d_I_expr * d_DC_expr * d_T_expr) / (CI_d * CT_d * CDC_d), name="d_def")

e_I_expr = eI2 * I * I + eI1 * I + eI0
e_DC_expr = eDC3 * DC * DC * DC + eDC2 * DC * DC + eDC1 * DC + eDC0
e_T_expr = eT3 * T**3 + eT2 * T**2 + eT1 * T + eT0
e = model.addVar(name="e")
CI_e, CT_e, CDC_e = cvals['e']
model.addConstr(e == (e_I_expr * e_DC_expr * e_T_expr) / (CI_e * CT_e * CDC_e), name="e_def")
model.update()


# In[449]:


# z = d·x + e
z = model.addVar(name="z")
model.addConstr(z == d * x + e)

# exp(z)
exp_z = model.addVar(name="exp_z")
model.addGenConstrExp(z, exp_z)

# c·exp(z)
c_exp_z = model.addVar(name="c_exp_z")
model.addConstr(c_exp_z == c * exp_z)

# V(x) = a·x + b + c·exp(z)
V_model = model.addVar(name="V_model")
model.addConstr(V_model == a * x + b + c_exp_z)

model.update()


# In[450]:


model.setObjective(x, GRB.MAXIMIZE)

# model.addConstr(V_soc_sq_f >= 0)
# model.addConstr(V_soc_sq_f <= 1)
model.addConstr(V_model <= 3, "vbatt_ub")
model.addConstr(V_model >= 2.4, name="min_voltage_cutoff")

model.update()


# In[451]:


model.write("model.lp")
model.setParam("OutputFlag", 1)


# In[454]:


#model.setObjective(x, GRB.MAXIMIZE)
model.optimize()

if model.Status == GRB.INFEASIBLE:
    model.computeIIS()
    model.write("model.ilp")


# In[417]:


# Print all key variable values after solving
if model.status == GRB.OPTIMAL or model.status == GRB.SUBOPTIMAL:
    print(f"x = {x.X}")
    print(f"f = {f.X}")
    print(f"I = {I.X}")
    print(f"V_model = {V_model.X}")
    print(f"P = {P.X}")
    print(f"DC = {DC.X}")
else:
    print("No feasible solution found.")


# In[ ]:





# In[ ]:




