from __future__ import division
from argparse import ArgumentParser
from visual import *
from socket import socket, AF_INET, SOCK_STREAM

HOST = 'localhost' # Symbolic name meaning all available interfaces
PORT = 50007 # Arbitrary non-privileged port

WIDTH  = 6.
LENGTH = 8.
HEIGHT = 3.

def update((x, y, z), (v_x, v_y, v_z), dt):
    dx, dy, dz = v_x*dt, v_y*dt, v_z*dt
    return x+dx, y+dy, z+dz
    

def generate_trajectory(dt=0.01):
    points = list()
    v = (0, 0, 0.2)
    init = (0, 0, 0)
    points.append(init)
    pos = init
    
    for i in xrange(int(5/dt)):
        pos = update(pos, v, dt)
        points.append(pos)

    v = (0.1, 0.1, 0.1)
    for i in xrange(int(50/dt)):
        x, y, z = pos
        vx, vy, vz = v
        if x >= WIDTH:
            vx = -vx
            
        if y >= LENGTH:
            vy = -vy
            
        if z >= HEIGHT:
            vz = -vz

        v = vx, vy, vz
        pos = update(pos, v, dt)
        points.append(pos)

    return points

def read_trajectory(filename):
    tfile = open(filename, 'r')
    trajectory = tfile.read().strip().split('\n')
    trajectory = filter(lambda x: x.startswith('3D'), trajectory)
    trajectory = [l.split(' = ')[-1][1:-1].split(', ') for l in trajectory]
    trajectory = [map(float,l) for l in trajectory]
    return trajectory

def transform_f2w(x, y, z):
    x_ = x + WIDTH/2
    y_ = y + HEIGHT/2
    z_ = -z + LENGTH/2

    return (x_, z_, y_)
    
def transform_w2f(x, y, z):
    x_ = x - WIDTH/2
    y_ = z - HEIGHT/2
    z_ = -y + LENGTH/2

    return (x_, y_, z_)

if __name__ == '__main__':
    parser = ArgumentParser(description='Process some integers.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", action="store", type=str)
    group.add_argument("--net", action="store", type=str)
    args = parser.parse_args()
    
    arena = box(pos=(0, 0, 0), width=LENGTH, height=HEIGHT, length=WIDTH, color=(0.6,0.6,0.6), opacity=0.2)
    copter = sphere(pos=transform_w2f(0,0,0), radius=0.05, color=color.red)
    clabel = label(pos=(0,HEIGHT+.2,0), text='This is a box', line=False)
    copter.trail = curve(color=(1, 0.2, 0.2))
    copter.velocity = vector(0.1,0.1,0.1)

    if args.net:
        sock = socket(AF_INET, SOCK_STREAM)
        sock.bind((HOST, PORT))
        sock.listen(1)
        conn, addr = sock.accept()
        print 'Connected by', addr
        while 1:
            data = conn.recv(1024)
            if not data: break
            data = map(float, data.split(' '))
            x, y, z, dt = data[0], data[1], data[2], data[3]
            rate (int(1000/dt))
            copter.pos = transform_w2f(x, y, z) #copter.pos + copter.velocity*dt
            copter.trail.append(pos=copter.pos)
            clabel.text = '%.2f, %.2f, %.2f' % transform_f2w(*tuple(copter.pos))
            
    else:
        dt = 35

        # points = generate_trajectory(dt)
        # points = read_trajectory(args.file)
        tfile = open(args.file, 'r')
        for l in tfile:
            if l.startswith('Time'):
                dt = float(l.split('=')[-1].split(' ')[0])
            rate (int(1000/dt))
            if l.startswith('3D'):           
                #l = l.split(' = ')[-1].split(', ')
                l = l.split('(', 1)[1].split(')')[0].split(', ')
                x = float(l[0])
                y = float(l[1])
                z = float(l[2])
                #print x,y,z
                copter.pos = transform_w2f(x, y, z) #copter.pos + copter.velocity*dt
                copter.trail.append(pos=copter.pos)
            clabel.text = '%.2f, %.2f, %.2f' % transform_f2w(*tuple(copter.pos))


"""
while 1:
    rate (100)
    x, y, z = copter.pos
    x, y, z = transform_f2w(x, y, z)
    dx, dy, dz = copter.velocity*dt
    x, y, z = x + dx, y + dy, z + dz
    copter.pos = transform_w2f(x, y, z) #copter.pos + copter.velocity*dt
    copter.trail.append(pos=copter.pos)
    clabel.text = '%.2f, %.2f, %.2f' % transform_f2w(*tuple(copter.pos))
"""
