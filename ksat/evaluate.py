#!/usr/bin/env python3.7
l='#'
k=open
j=str
T='user'
S='sat'
R=''
Q=float
M='Invalid line! '
L='interferers'
K=range
I='sats'
H=True
F='users'
D=len
B=False
A=print
import sys as G
from collections import namedtuple as U
from math import sqrt as N,acos,degrees as V,floor
C=U('Vector3',['x','y','z'])
W=C(0,0,0)
X=32
Y=[j(A)for A in K(1,X+1)]
Z=4
a=[chr(ord('A')+A)for A in K(0,Z)]
b=10.0
c=20.0
O=45.0
def J(vertex,point_a,point_b):
	G=point_b;F=point_a;B=vertex;D=C(F.x-B.x,F.y-B.y,F.z-B.z);E=C(G.x-B.x,G.y-B.y,G.z-B.z);H=N(D.x**2+D.y**2+D.z**2);I=N(E.x**2+E.y**2+E.z**2);J=C(D.x/H,D.y/H,D.z/H);K=C(E.x/I,E.y/I,E.z/I);L=J.x*K.x+J.y*K.y+J.z*K.z;M=min(1.0,max(-1.0,L))
	if abs(M-L)>1e-06:A(f"dot_product: {L} bounded to {M}")
	return V(acos(M))
def d(scenario,solution):
	O=solution;L=scenario;A('Checking no sat interferes with itself...')
	for M in O:
		C=O[M];E=list(C.keys());Q=L[I][M]
		for G in K(D(C)):
			for N in K(G+1,D(C)):
				R=C[E[G]][1];S=C[E[N]][1]
				if R!=S:continue
				T=C[E[G]][0];U=C[E[N]][0];V=L[F][T];W=L[F][U];P=J(Q,V,W)
				if P<b:A(f"\tSat {M} beams {E[G]} and {E[N]} interfere.");A(f"\t\tBeam angle: {P} degrees.");return B
	A('\tNo satellite self-interferes.');return H
def e(scenario,solution):
	E=solution;C=scenario;A('Checking no sat interferes with a non-Starlink satellite...')
	for D in E:
		N=C[I][D]
		for G in E[D]:
			O=E[D][G][0];P=C[F][O]
			for K in C[L]:
				Q=C[L][K];M=J(P,N,Q)
				if M<c:A(f"\tSat {D} beam {G} interferes with non-Starlink sat {K}.");A(f"\t\tAngle of separation: {M} degrees.");return B
	A('\tNo satellite interferes with a non-Starlink satellite!');return H
def f(scenario,solution):
	C=solution;A('Checking user coverage...');E=[]
	for I in C:
		for K in C[I]:
			G=C[I][K][0]
			if G in E:A(f"\tUser {G} is covered multiple times by solution!");return B
			E.append(G)
	J=D(scenario[F]);L=D(E);A(f"{L/J*100}% of {J} total users covered.");return H
def g(scenario,solution):
	E=scenario;D=solution;A('Checking each user can see their assigned satellite...')
	for C in D:
		for L in D[C]:
			G=D[C][L][0];M=E[F][G];N=E[I][C];K=J(M,W,N)
			if K<=180.0-O:P=j(K-90);A(f"\tSat {C} outside of user {G}'s field of view.");A(f"\t\t{P} degrees elevation.");A(f"\t\t(Min: {90-O} degrees elevation.)");return B
	A("\tAll users' assigned satellites are visible.");return H
def E(object_type,line,dest):
	F=line;E=F.split()
	if E[0]!=object_type or D(E)!=5:A(M+F);return B
	else:
		G=E[1]
		try:I=Q(E[2]);J=Q(E[3]);K=Q(E[4])
		except:A("Can't parse location! "+F);return B
		dest[G]=C(I,J,K);return H
def h(filename,scenario):
	K='interferer';G=filename;D=scenario;A('Reading scenario file '+G);J=k(G).readlines();D[I]={};D[F]={};D[L]={}
	for C in J:
		if l in C:continue
		elif C.strip()==R:continue
		elif K in C:
			if not E(K,C,D[L]):return B
		elif S in C:
			if not E(S,C,D[I]):return B
		elif T in C:
			if not E(T,C,D[F]):return B
		else:A(M+C);return B
	return H
def P(filename,scenario,solution):
	P=scenario;L=filename;K=solution
	if L==R:A('Reading solution from stdin.');N=G.stdin
	else:A(f"Reading solution file {L}.");N=k(L)
	V=N.readlines()
	for E in V:
		C=E.split()
		if l in E:continue
		elif D(C)==0:continue
		elif D(C)==8:
			if C[0]!=S or C[2]!='beam'or C[4]!=T or C[6]!='color':A(M+E);return B
			J=C[1];O=C[3];Q=C[5];U=C[7]
			if not J in P[I]:A('Referenced an invalid sat id! '+E);return B
			if not Q in P[F]:A('Referenced an invalid user id! '+E);return B
			if not O in Y:A('Referenced an invalid beam id! '+E);return B
			if not U in a:A('Referenced an invalid color! '+E);return B
			if not J in K:K[J]={}
			if O in K[J]:A('Beam is allocated multiple times! '+E);return B
			K[J][O]=Q,U
		else:A(M+E);return B
	N.close();return H
def i():
	if D(G.argv)!=3 and D(G.argv)!=2:A('Usage: python3.7 evaluate.py /path/to/scenario.txt [/path/to/solution.txt]');A('   If the optional /path/to/solution.txt is not provided, stdin will be read.');return-1
	B={}
	if not h(G.argv[1],B):return-1
	C={}
	if D(G.argv)!=3:
		if not P(R,B,C):return-1
	elif not P(G.argv[2],B,C):return-1
	if not f(B,C):return-1
	if not g(B,C):return-1
	if not d(B,C):return-1
	if not e(B,C):A('Solution contained a beam that could interfere with a non-Starlink satellite.');return-1
	A('\nSolution passed all checks!\n');return 0
if __name__=='__main__':exit(i())