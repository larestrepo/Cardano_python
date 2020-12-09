import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
import pandas
import argparse
from urllib.request import urlopen
import json
from sklearn.linear_model import LinearRegression
import matplotlib.patches as mpatches

parser = argparse.ArgumentParser(description="Calculate probability of minting block given parameters")
parser.add_argument('--d-param', dest='d', type=float, help='the current decentralization parameter [e.g. 0.0 - 1.0]')
parser.add_argument('--epoch', dest='epoch', type=int, help='the epoch number [e.g. 221]')
parser.add_argument('--Ps', dest='Ps', type=float, help='Pool active stake in ADA')
parser.add_argument('--Os', dest='Os', type=float, help='Pool pledge in ADA')
parser.add_argument('--Vc', dest='Vc', type=float, help='Pool variable cost in %')
parser.add_argument('--Fc', dest='Fc', type=float, help='Pool fixed cost in ADA')

""" #Setting up parameters
Blocks: 21600 Total blocks per epoch
ro: Monetary expansion param
a0: pledge influence factor
tau: fraction of the reward going to the treasury
k: number of desired pools
Tm: Maximum ADA supply
T: Total ADA supply
Ts:Total Active stake
Ps=1820000 Pool Active stake
Os=70000 Owner address stake
"""
#Blockchain parameters hardcoded
Blocks=21600
a0=0.3
ro=0.003
tau=0.2
k=500
Tm=45000000000
T=31900000000
Ts=19910000000

args = parser.parse_args()

epoch = args.epoch
if epoch == None:
   print("\033[94m[INFO]:\033[0m No epoch provided, using latest known epoch.")
   url=("https://epoch-api.crypto2099.io:2096/epoch/")
else:
   url=("https://epoch-api.crypto2099.io:2096/epoch/"+str(epoch))

try:
    page = urlopen(url)
    epoch_data = json.loads(page.read().decode("utf-8"))
    print("epoch: ",epoch)
except:
    print("\033[1;31m[WARN]:\033[0m Unable to fetch data from the epoch API.")


try:
    d = args.d or epoch_data['d']
    print("Decentralization param: ",d)
except:
    print("\033[1;31m[ERROR]:\033[0m One or more arguments are missing or invalid. Please try again.")
    parser.format_help()
    parser.print_help()
    exit()

Ps=args.Ps
if Ps == None:
   print("\033[94m[INFO]:\033[0m No active stake provided, using default=1'680.000 ADA.")
   Ps=1680000
Os=args.Os
if Os == None:
   print("\033[94m[INFO]:\033[0m No pool pledge provided, using default=45.400 ADA.")
   Os=45400
Vc=args.Vc
if Vc == None:
   print("\033[94m[INFO]:\033[0m No varible cost provided, using default=1%.")
   Vc=0.01
Fc=args.Fc
if Fc == None:
   print("\033[94m[INFO]:\033[0m No fixed cost provided, using default=340 ADA.")
   Fc=340

#Pool parameters
#fixed_cost=340
#var_cost=0.01

"""Calculations of the blockchain"""
#blocks per epoch
blocks_epoch=Blocks*(1-d)
#Saturation point
z0=1/k 
#Total rewards for everyone in the blockchain
Re=ro*(Tm-T) 
#Total rewards after treasury tax
R=Re*(1-tau) 

"""Calculations of the pool"""
#Pool active stake/total stake
sigma_a=Ps/Ts 
#Expected # of blocks produced per epoch
block=sigma_a*blocks_epoch 
#Pool active stake/Total supply
sigma=Ps/T 
#Owner active stake/total supply
s=Os/T 
#Apparent performance of the pool
Performance=block/blocks_epoch/sigma_a
#Saturation effects
sigma_prime=min(sigma,z0)
s_prime=min(s,z0)
#Rewards to the pool
Rp=R/(1+a0)*(sigma_prime+(s_prime*a0)*(sigma_prime-(s_prime*(z0-sigma_prime)/z0))/z0) 
#Real rewards to the pool
Rc=Rp*Performance
#Rewards per block
Rb=Rc/block 
#Rewards to delegators
Rd=(Rc-Fc)*(1-(Vc/100))
if Rd < 0:
   Rd=0
#Anualized ROS
ROA=Rd/Ps*73*100

print("sigma: ",sigma_a,"\n""expected # blocks: ",block,"\n""Total rewards blockchain: ",Re,"\n""Total rewards after treasury: ",R,
"\n""Pool performance: ",Performance,"\n""Rewards to the pool: ",Rp,"\n""Rewards after performance: ",Rc,"\n""Rewards per block",Rb,
"\n""Rewards to delegators: ",Rd,"\n""ROA: ",ROA)

dist=np.random.binomial(blocks_epoch,sigma_a,1000)
dist_array=np.array(dist)
dist_series=pandas.DataFrame(dist_array)
fx,ax=plt.subplots()
l1=dist_series.plot.kde(ax=ax)
l2=dist_series.plot.hist(density=True,ax=ax,bins=10, grid=True,rwidth=0.9)
#Formatting the plot
plt.title("Cálculos y probabilidades de minar")
plt.xlabel("# de bloques estimados", fontsize=15)
plt.ylabel("% de probabilidad", fontsize=15)
red_patch = mpatches.Patch(label='PDF')
plt.legend(handles=[red_patch])
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.show()

# Group of functions for different scenarios
def stake_vs_block():
   Ps_var=[]
   block_stake_var=[]
   stake=Ps
   while stake <=100000000:
      stake=stake+100000
      block_stake=stake/Ts*blocks_epoch
      Ps_var.append(stake)
      block_stake_var.append(block_stake)
   return [Ps_var,block_stake_var]

def K_vs_rewards():
   z0_var=[]S
   Rp_var=[]
   for k in range(1,1000):
      z0=1/k
      z0_var.append(k)
      sigma_prime=min(sigma,z0)
      s_prime=min(s,z0)
      Rp=R/(1+a0)*(sigma_prime+(s_prime*a0)*(sigma_prime-(s_prime*(z0-sigma_prime)/z0))/z0) 
      Rp_var.append(Rp)
   return [z0_var,Rp_var]

def a0_vs_rewards():
   a0_var=[]
   Rp_var=[]
   for a0 in np.arange(1,100,0.01):
      z0=1/k
      a0_var.append(a0)
      sigma_prime=min(sigma,z0)
      s_prime=min(s,z0)
      Rp=R/(1+a0)*(sigma_prime+(s_prime*a0)*(sigma_prime-(s_prime*(z0-sigma_prime)/z0))/z0) 
      Rp_var.append(Rp)
   return [a0_var,Rp_var]

def ROS_vs_stake():
   Ps_var=[]
   ROS_var=[]
   stake=50000
   while stake <=100000000:
      stake=stake+10000
      sigma_a=stake/Ts
      block=sigma_a*blocks_epoch
      sigma=stake/T
      Performance=block/blocks_epoch/sigma_a
      sigma_prime=min(sigma,z0)
      s_prime=min(s,z0)
      Rp=R/(1+a0)*(sigma_prime+(s_prime*a0)*(sigma_prime-(s_prime*(z0-sigma_prime)/z0))/z0) 
      Rc=Rp*Performance
      Rd=(Rc-Fc)*(1-(Vc/100))
      if Rd < 0:
         Rd=0
      ROA=Rd/stake*73*100
      Ps_var.append(stake)
      ROS_var.append(ROA)
   return [Ps_var,ROS_var]

def ROS_vs_a0():
   a0_var=[]
   ROS_var=[]
   stake=Ps
   for a0 in np.arange(0,100,0.01):
      z0=1/k
      a0_var.append(a0)
      Performance=block/blocks_epoch/sigma_a
      sigma_prime=min(sigma,z0)
      s_prime=min(s,z0)
      Rp=R/(1+a0)*(sigma_prime+(s_prime*a0)*(sigma_prime-(s_prime*(z0-sigma_prime)/z0))/z0) 
      Rc=Rp*Performance
      Rd=(Rc-Fc)*(1-(Vc/100))
      if Rd < 0:
            Rd=0      
      ROA=Rd/stake*73*100
      ROS_var.append(ROA)
   return [a0_var,ROS_var]

#Case 1: increase in stake maintaning blockchain parameters constant
shape=stake_vs_block()
X=np.array(shape[0]).reshape((-1,1))
Y=np.array(shape[1])
# Apply linear regression
regressor=LinearRegression()
regressor.fit(X,Y)
r_sq=regressor.score(X,Y)
intercept=regressor.intercept_
slope=regressor.coef_
print("R square",r_sq,"\n""intercept",intercept,"\n""slop",slope,"\n""Conclusión Caso 1: ","El número de bloques es directamente proporcional al número", 
"al stake de manera lineal.","\n""Cada 0.7M de ADAs aumenta la posibilidad de minar 1 bloque manteniendo los otros parámetros constantes")
plt.scatter(X,Y,color='red')
plt.plot(X,regressor.predict(X), color='blue')
plt.title("Stake-vs-# de Bloques")
plt.xlabel("Stake en millones de ADAs", fontsize=15)
plt.ylabel("# de bloques", fontsize=15)
plt.show()

#Case 2: impact on k over the stake pool rewards
shape=K_vs_rewards()
print("Conclusión Caso 2: ","Con los parámetros proporcionados para el pool la saturación comienza en el punto de quiebre de la curva")
plt.scatter(shape[0],shape[1])
plt.title("K-vs-recompensas""\n""Con los parámetros proporcionados para el pool""\n""la saturación comienza en el punto de quiebre de la curva")
plt.xlabel("K de 0 a 1000", fontsize=15)
plt.ylabel("Recompensas", fontsize=15)
plt.show()


#Case 3: impact on pledge factor over the stake pool rewards
shape=a0_vs_rewards()
plt.scatter(shape[0],shape[1])
print("Conclusión Caso 3: ","Poca variación en el parámetro a0 impacta fuertemente en las recompensas")
plt.title("a0 o influencia del pledge-vs-recompensas""\n""Poca variación en el parámetro a0 impacta fuertemente en las recompensas")
plt.xlabel("Parámetro a0", fontsize=15)
plt.ylabel("Recompensas", fontsize=15)
plt.show()

#Case 4: ROS vs stake influence
shape=ROS_vs_stake()
plt.scatter(shape[0],shape[1])
print("Conclusión Caso 4: ","Los pooles con un aumento pequeño de stake rápidamente alcanzan un ROS de 5%. Este disminuye en el punto de saturación")
plt.title("stake-vs-ROS""\n""Los pooles con un aumento pequeño de stake rápidamente alcanzan un ROS de 5%.""\n""Este disminuye en el punto de saturación")
plt.xlabel("Stake en millones de ADAs", fontsize=15)
plt.ylabel("ROS", fontsize=15)
plt.show()


#Case 5: ROS vs a0
shape=ROS_vs_a0()
plt.scatter(shape[0],shape[1])
print("Conclusión Caso 5: ","Poca variación en el parámetro a0 impacta fuertemente el ROS")
plt.title("a0 o influencia del pledge-vs-ROS""\n""Poca variación en el parámetro a0 impacta fuertemente el ROS")
plt.xlabel("a0", fontsize=15)
plt.ylabel("ROS", fontsize=15)
plt.show()