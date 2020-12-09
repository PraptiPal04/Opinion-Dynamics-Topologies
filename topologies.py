import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib as mpl
import matplotlib.animation as an

def rk4(f,x0,T,h,N,*args,**kwargs):
    '''
    To perform integration stepwise using the Runge-Kutta stage 4 Method

    Parameters
    ----------
    f : function
        to calculate the RHS of the ode
    x0 : numpy array
        initial condiiton of the system
    T : float
        the time interval
    h : float
        the time step
    *args : additional arguments to pass down to the function f
    **kwargs : additional keyword arguments to pass down to the function f

    Returns
    -------
    x : numpy array
        calculated values of the system at each step

    '''
    x=np.zeros((N,int((T/h)+1)))
    x[:,0]=x0
    for i in range(int(T/h)):
        k1=f(x[:,i],*args,**kwargs)
        k2=f(x[:,i]+(h/2)*k1,*args,**kwargs)
        k3=f(x[:,i]+(h/2)*k2,*args,**kwargs)
        k4=f(x[:,i]+h*k3,*args,**kwargs)
        x[:,i+1] = x[:,i] + (h/6)*(k1+2*k2+2*k3+k4)
    return x

def rhs(x,A,d,u,al,gm,b):
    '''
    TO calculate the RHS of the given ode

    Parameters
    ----------
    x : numpy array
        curent state of the system
    A : numpy array
        the adjacency matrix of the system graph
    d : float
        parameter, resistance to becoming opinionated
    u : float
        control parameter, social influence
    al : float
        parameter, self reinforcement
    gm : float
        parameter, cooperative/competitive
    b : float
        parameter, input bias

    Returns
    -------
    x_dot : numpy array
        RHS of the ode. The derivative of the current state.

    '''
    x1=np.zeros(np.shape(x))
    for i in range(np.shape(A)[0]):
        for j in range(np.shape(A)[1]):
            if (i!=j):
                x1[i]+=gm*A[i,j]*x[j]
            else:
                x1[j]+=al*x[j]
    x_dot=-d*x+u*np.tanh(x1)+b
    return x_dot

N=10 #number of nodes

A=np.zeros((N,N))   #adjacency matrix

x0=(np.random.uniform(size=N)-0.5)*2    #initial state

#TOPOLOGIES

# #Circle 
# for i in range(N-1):
#     A[i,i+1]=1
#     A[i+1,i]=1
    
#     A[N-1,0]=1
#     A[0,N-1]=1

#Wheel
for i in range(N-2):
    A[i,i+1]=1
    A[i+1,i]=1
    A[i,N-1]=1
    A[N-1,i]=1
A[N-2,N-1]=1
A[N-1,N-2]=1
A[N-2,0]=1
A[0,N-2]=1

# #Path
# for i in range(N-1):
#     A[i,i+1]=1
#     A[i+1,i]=1

# #Star
# for i in range(N-2):
#     A[i,N-1]=1
#     A[N-1,i]=1



rows,cols=np.where(A==1.)
edges=zip(rows.tolist(),cols.tolist())
G=nx.Graph()
G.add_edges_from(edges)
plot_positions=nx.drawing.spring_layout(G)

vmin=-2
vmax=2
norm=mpl.colors.Normalize(vmin=vmin,vmax=vmax)
cmap=plt.get_cmap('coolwarm')
sm=plt.cm.ScalarMappable(cmap=cmap,norm=norm)
sm.set_array([])

#Plotting

x=rk4(rhs,x0,15.0,0.05,N,A,d=1.0,u=0.26,al=1.2,gm=-1.3,b=0.0)
# t=np.arange(0.,15.05,0.05)
# for i in range(N):
#     plt.plot(t,x[i,:])
# plt.title("Wheel Topology Disagreement")
# plt.ylabel("$x(t)$")
# plt.xlabel("$t$")
# plt.savefig("Wheel Disagreement.png")
# plt.show()

#Animation. Save video

fig=plt.gcf()

def animate(i):
    '''
    Function to iterate over in order to animate the graphs

    Parameters
    ----------
    i : int
        interative variable for the FuncAnimation() function to animate the graph

    Returns
    -------
    None.

    '''
    nx.draw(G,pos=plot_positions,node_size=500,node_color=x[:,i],cmap='coolwarm',vmin=vmin,vmax=vmax)


plt.colorbar(sm)
anim = an.FuncAnimation(fig, animate, frames=200, blit=False)
writervideo = an.FFMpegWriter(fps=10) 
anim.save('WHeel Topology Disagreement.mp4', writer=writervideo)


