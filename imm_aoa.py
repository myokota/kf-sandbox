#!/usr/bin/env python
# coding: utf-8

import copy
from math import sin, cos, tan, atan2, sqrt
from filterpy.kalman import KalmanFilter
from filterpy.kalman import IMMEstimator
from robot_aoa import *
from scipy.linalg import block_diag
import numpy as np

class EstimationAgent(Agent):
    def __init__(self, time_interval, cmds, estimator):
        super().__init__(0, 0)
        self.estimator = estimator
        self.time_interval = time_interval

        self.prev_nu = 0
        self.prev_omega = 0

        self.cmds = cmds
        self.cmd_step = 0

    # observasion: data from the sensors
    # self.{nu,omega}: movement
    def decision(self, observation=None):
        self.estimator.motion_update(self.prev_nu, self.prev_omega, self.time_interval)

        if len(self.cmds) == 0:
            self.cmds.append([1, 0, 0])

        steps, nu, omega = self.cmds[0]
        if steps < self.cmd_step:
            del cmds[0]
            self.cmd_step = 0

        self.cmd_step += 1

        self.prev_nu, self.prev_omega = nu, omega
        self.estimator.observation_update(observation)
        return nu, omega

    def draw(self, ax, elems): ###mlwrite
        self.estimator.draw(ax, elems)
        #x, y, t = self.estimator.pose #以下追加
        #s = "({:.2f}, {:.2f}, {})".format(x,y,int(t*180/math.pi)%360)
        #elems.append(ax.text(x, y+0.1, s, fontsize=8))

def line_cross_point(P0, P1, Q0, Q1):
    x0, y0 = P0; x1, y1 = P1
    x2, y2 = Q0; x3, y3 = Q1
    a0 = x1 - x0; b0 = y1 - y0
    a2 = x3 - x2; b2 = y3 - y2

    d = a0*b2 - a2*b0
    if d == 0:
        # two lines are parallel
        return None

    # s = sn/d
    sn = b2 * (x2-x0) - a2 * (y2-y0)
    # t = tn/d
    #tn = b0 * (x2-x0) - a0 * (y2-y0)
    return x0 + a0*sn/d, y0 + b0*sn/d

class EstimatorIMME:
    def __init__(self, envmap, dt, init_pose):

        r =1.0
        ca = KalmanFilter(6, 2)
        dt2 = (dt**2)/2
        F = np.array([[1, dt, dt2],
                      [0,  1,  dt],
                      [0,  0,   1]])

        ca.F = block_diag(F, F)
        #ca.x = np.array([init_pose[0], 0, 0, init_pose[1], -15, 0]).T
        ca.x = np.array([init_pose[0], 0, 0, init_pose[1], 0, 0]).T
        ca.P *= 1.e-12
        ca.R *= r**2
        q = np.array([[.05, .125, 1/6],
                      [.125, 1/3, .5],
                      [1/6, .5, 1]])*1.e-3
        ca.Q = block_diag(q, q)
        ca.H = np.array([[1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0]])

        # プロセスノイズを持たない同じフィルタを作成する。
        cano = copy.deepcopy(ca)
        cano.Q *= 0
        filters = [ca, cano]

        M = np.array([[0.97, 0.03],
                      [0.03, 0.97]])
        mu = np.array([0.5, 0.5])
        bank = IMMEstimator(filters, mu, M)

        self.bank = bank
        self.pose = init_pose
        self.estimated_poses = []
        self.antennas = envmap.antennas


    def motion_update(self, nu, omega, time): #追加
        #値が0になるとゼロ割りになって計算ができないのでわずかに値を持たせる
        if abs(omega) < 1e-5:
            omega = 1e-5

        self.bank.predict()

        #  pose = [x, y, theta]
        pose = np.array([self.bank.x[0], self.bank.x[3], self.pose[2]]).T
        self.pose = IdealRobot.state_transition(nu, omega, time, pose)

    def observation_update(self, observation):  #追加
        # observation[]: [(0, 角度), アンテナ位置, 観測ID]
        #    z = d[0][1] # d[0] = (0, theta)
        #    ant_pos = d[1]
        #    obs_id = d[2]
        ant1 =observation[0]
        ant2 =observation[1]
        z1 = ant1[0][1]
        z2 = ant2[0][1]
        a1 = math.tan(z1)
        a2 = math.tan(z2)
        p1 = ant1[1]
        p2 = ant2[1]
        x1 = 10
        x2 = 0
        c = line_cross_point((p1[0], p1[1]), (x1, a1*x1+a1*-p1[0]+p1[1]),
                             (p2[0], p2[1]), (x2, a2*x2+a2*-p2[0]+p2[1]))
        self.bank.update([c[0], c[1]])

    def draw(self, ax, elems):
        x = [self.bank.x[0], self.bank.x[3], 0]

        #elems.append(ax.text(0.5, 9.0, "x:" + str(x[0]), fontsize=10))
        #elems.append(ax.text(0.5, 8.5, "y:" + str(x[1]), fontsize=10))
        #elems += ax.plot(x[0], x[2], color="blue", alpha=0.5, linewidth=0.5, marker='o')
        self.estimated_poses.append(x)
        xs = [_x[0] for _x in self.estimated_poses]
        ys = [_x[1] for _x in self.estimated_poses]
        elems += ax.plot(xs, ys, color="blue", alpha=0.5, linewidth=0.5)
        c = patches.Circle(xy=(x[0], x[1]), radius=0.1, fill=True, alpha=0.5, color="blue")
        elems.append(ax.add_patch(c))


# In[10]:

if __name__ == '__main__':
    dt = 0.1
    #world = World(60, dt, debug=True)
    world = World(60, dt)

    m = Map()
    m.append_antenna(Antenna(0.5, 0.5))
    m.append_antenna(Antenna(9.5, 0.5))
    world.append(m)

    # cmds = [[steps, nu, omega] * n]
    cmds = [[100, 0.4, 0.0],
            [100, 0.2, 10.0],
            [100, 0.4, 0.0],
            [100, 0.2, 10.0],
            [100, 0.0, 0.0],
            [100, 0.4, 0.0]]
    cmds = [[s, n, math.radians(o)] for s, n, o in cmds]

    estimator_pose = np.array([5, 5, 0])
    robot_pose = np.array([2, 2, 0])
    e = EstimatorIMME(m, dt, estimator_pose)
    a = EstimationAgent(dt, cmds, e)
    r = Robot(robot_pose, sensor=AoA(m), agent=a, color="red")

    world.append(r)

    ### アニメーション実行 ###
    world.draw()

