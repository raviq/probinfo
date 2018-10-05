import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import NullFormatter, NullLocator, MultipleLocator
from string import ascii_uppercase, ascii_lowercase, digits
from bisect import bisect
from random import randint, random, uniform, choice
import entropy_estimators as ee

def kldiv(p, q):
	p = np.asarray(p, dtype=np.float)
	q = np.asarray(q, dtype=np.float)
	return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def banana_distribution(N):
    # Generate random points in a banana shape
    # create a truncated normal distribution
    theta = np.random.normal(0, np.pi / 8, N)
    theta[theta >= np.pi / 4] /= 2
    theta[theta <= -np.pi / 4] /= 2
    # define the curve parametrically
    r = np.sqrt(1. / abs(np.cos(theta) ** 2 - np.sin(theta) ** 2))
    mu, sigma = 0.3, 0.08
    r += np.random.normal(mu, sigma, size=N)
    x = r * np.cos(theta + np.pi / 4)
    y = r * np.sin(theta + np.pi / 4)
    return (x, y)

def process(A, P, T):
	# generates a sequence of length n, from alphabet A, with prob P
	cdf = [P[0]]
	for i in range(1, len(P)):
		cdf.append(cdf[-1] + P[i])
	return [A[bisect(cdf, random())] for t in range(T)]

def symbol(A, P):
	# return a symbol from alphabet A, with prob P
	cdf = [P[0]]
	for i in range(1, len(P)):
		cdf.append(cdf[-1] + P[i])
	return A[bisect(cdf, random())]

def gaussian(x, mu, sig):
    m = len(x)
    x = np.linspace(0, m-1, m)
    g = 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)
    return g/sum(g)

# plot the HMM
def plot_HMM(X, T, m, n, mu, sigma, c, p0, P, Q):

	# X: sample space
	# T: horizon
	# m: number of observable features
	# n: number of hidden states
	# p0: bootstrap prob, to select the first state
	# P, mu[], sigma[]: P, Gaussian over observations O
	# Q: Transition prob between the n hidden states
	# c: colors

	O = range(m) # observable features (O)
	S = range(n) # states  
	colors = 'rbmgyk'
	
	fig, axs = plt.subplots(figsize=(12, 6), facecolor="white")
	ax0 = plt.axes((0.05, 0.1, 0.24, 0.7))

	for j in range(n):
		ax0.plot(P[j], O, colors[j]+'-', label='$p_%d$' % j)	
 		plt.axhline(y=mu[j], xmin=0, xmax=1., color=colors[j], linestyle='-', linewidth=.3)
 
	ax0.xaxis.set_major_formatter(NullFormatter())
	ax0.set_ylabel(r'$o$')
	ax0.set_xlabel(r'$p_j(o)\sim N(\mu_j, \sigma_j)$')
	ax0.xaxis.set_label_position('bottom')
	plt.title('State-distributions')
	#plt.legend(loc=7)
	plt.grid()

	
	ax1 = plt.axes((0.299, 0.1, 0.62, 0.7))
	
	for j in range(n):
		plt.axhline(y=mu[j], xmin=0, xmax=1., color=colors[j], linestyle='-', linewidth=.3)
	
		print ' (%s) P_%d = %s %.1f   %f\t %s' % (colors[j], j, Q[j], sum(Q[j]), ee.entropyfromprobs(Q[j]), c.count(colors[j])/(T*1.) )
		#print ''
		#print sum([Q[_][j] for _ in range(n)])

	for t in range(T):
		ax1.scatter(t, X[t], c=c[t], marker='s', edgecolors='none', s=10)
	plt.plot(X, 'k-', lw=.05)
	
	

	
	ax1.set_xlim(0, T)
	ax1.set_xlabel('$t$')
	ax1.yaxis.set_major_formatter(NullFormatter())
	ax1.set_ylim(0, m)
	plt.grid()
	
	axs.axis('off')
	plt.title('HMM Sample Space')
	fig.savefig('figures/HMM.png', dpi=1000)

#--------------------------------------------------------------------------------

T = 1000	# Horizon length
setting = 'HMM'

N = range(T)
E = [0] * T
S = [0] * T	

# Alphabets
#oE = range(Ngrid)	# Ngrid: Number of variables/observables
#oS = range(Ngrid) #ascii_uppercase[:Ngrid]
O = range(3)

if setting == 'Independent':
	px_ = np.random.dirichlet(np.ones(Ngrid), size=1)[0]		
	py_ = np.random.dirichlet(np.ones(Ngrid), size=1)[0]		
	E = process(oE, px_, T) 
	S = process(oS, py_, T)

if setting == 'Delay':
	px_ = np.random.dirichlet(np.ones(Ngrid), size=1)[0]		
	E = process(oE, px_, T) 

	tau = 4
	S = [ oS[E[i+tau]] for i in range(T-tau) ]

elif setting == 'Joint':
	E = process(oE, px, T) 
	S = process(oS, py, T)
	
	
elif setting == 'HMM':

	#-----------------------------------
	# init
	#-----------------------------------
	m = 20 		 # number of observable features
	O = range(m) # observable features (O)	
	n = 3 		 # number of hidden states
	States = range(n) # states  
	# bootstrap probability
	p0 = np.random.dirichlet(np.ones(n), size=1)[0]
	
	# pick random state	
	current = symbol(States, p0)	

	# (1) states gaussian distributions over observations (P)
	mu = [4, 7, 13, 17, 2]	
	sigma = [2, 1, 3, 1, 1]
	P = [ gaussian(O, mu[j], sigma[j]) for j in range(n)]

	# first realisations
	E[0] = symbol(O, P[current])		# using p[current], sample and observation
	S[0] = choice(ascii_lowercase)
	
	# (2) Transition prob between hidden states (Q)
	alphas = np.ones(n)
	Q = np.random.dirichlet(alphas, size=n)	

	c = ['k'] * T
	colors = 'rbmgyk'
	
	for t in range(1, T):

		next = symbol(States, Q[current])		
		# sample from next state
		E[t] = symbol(O, P[next])
		current = next
		
		c[t] = colors[next]

		
		S[t] = ascii_uppercase[E[t-1]] if t<2 else ascii_lowercase[int(E[t-2])]
		
# 		if t < 2000:
# 			S[t] = choice(ascii_lowercase[:10])
# 		if t > 2000 and t < 6000:
# 			S[t] = choice(ascii_uppercase[:10])
# 		if t >= 8000:
# 			S[t] = choice(digits[:10])
			
		
		
		#S[t] = ascii_uppercase[E[t-1]]# if (random() < .5) else S[t-1]
		
		
# 		if t == 1000:
# 			# alter (1)
# 			mu = [1, 10, 15, 2, 18]	
# 			sigma = [3, 1, 1, 1, 1]
# 			p = [ gaussian(O, mu[j], sigma[j]) for j in range(n)]
# 			# alter (2)
# 			p_ = [np.random.dirichlet(np.ones(n), size=1)[0]	for j in range(n)]
	plot_HMM(E, T, m, n, mu, sigma, c, p0, P, Q)
	print 'E: %s' % ''.join( [ '%2s ' % _ for _ in E[:50]] )
	print 'S: %s' % ''.join( [ '%2s ' % _ for _ in S[:50]] )


else:
	pass
	
	

"""
fig, axs = plt.subplots(figsize=(12, 6), facecolor="white")

ax1 = plt.subplot(211)
ax1.plot(px, 'b.-', label='$p(x)$')
ax1.set_xticks(range(Ngrid))
ax1.set_xticklabels(oE) 
plt.grid()
plt.legend()
plt.xlabel('$x$')
plt.title('Environment (%s)' % setting)

ax2 = plt.subplot(212)
ax2.plot(py, 'm.-', label='$p(y)$')
ax2.set_xticks(range(Ngrid))
ax2.set_xticklabels(oS) 
plt.grid()
plt.legend()
plt.xlabel('$y$')
plt.title('Agent (%s)' % setting)
fig.savefig('figure2.png', dpi=1000)
"""
#------------------------------------------------------------------------------------------------


tau = 2
N = range(10, T-tau, 100)

A = [0] * len(N)
AA = [0] * len(N)
B = [0] * len(N)
C = [0] * len(N)
D = [0] * len(N)

for i, n in enumerate(N):

	A[i] = ee.centropyd(S[ :n+tau], S[ :n]) # H(S_{n+tau} | S_n) 
	AA[i] = ee.centropyd(S[ :n+tau], E[ :n]) # H(S_{n+tau} | E_n) 
	

	B[i] = ee.midd(S[ :n+tau], E[ :n])	# I( S_{n+tau}; E_n )
	C[i] = ee.midd(S[ :n], E[ :n])		# I( S_n | E_n )

	D[i] = ee.cmidd(S[ :n+tau], E[ :n], S[ :n])		# I( S_{n+tau}; E_n | S_n )


fig, axs = plt.subplots(figsize=(12, 6), facecolor="white")

plt.plot(N, A, 'b.-',label=r'$H(S_{n+\tau}\ |\ S_n)$')
plt.plot(N, AA, 'y-',label=r'$H(S_{n+\tau}\ |\ E_n)$')



plt.plot(N, B, 'r.-', label=r'$I(S_{n+\tau}\ ;\ E_n)$')
plt.plot(N, C, 'm-', label=r'$I( S_n ; E_n )$')

plt.plot(N, D, 'g.', label=r'$I(S_{n+\tau};\ E_n\ |\ S_n )$')

plt.legend(loc=7)
plt.xlabel('$n$')
plt.title(r'$\tau=%d,\ %s$' % (tau, setting) )
fig.savefig('figures/Information.png', dpi=1000)

print 'done.'




