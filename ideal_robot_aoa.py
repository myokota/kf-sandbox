#!/usr/bin/env python
# coding: utf-8

# In[4]:


import matplotlib
import matplotlib.animation as anm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np


# In[5]:


class World:
    def __init__(self, time_span, time_interval, debug=False):
        self.objects = []
        self.debug = debug
        self.time_span = time_span
        self.time_interval = time_interval

    def append(self,obj):
        self.objects.append(obj)

    def draw(self):
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_xlim(0,10)
        ax.set_ylim(0,10)
        ax.set_xlabel("X",fontsize=10)
        ax.set_ylabel("Y",fontsize=10)

        elems = []

        if self.debug:
            for i in range(int(self.time_span/self.time_interval)): self.one_step(i, elems, ax)
        else:
            self.ani = anm.FuncAnimation(fig, self.one_step, fargs=(elems, ax),
                                     frames=int(self.time_span/self.time_interval)+1, interval=int(self.time_interval*1000), repeat=False)
            plt.show()

    def one_step(self, i, elems, ax):
        while elems: elems.pop().remove()
        time_str = "t = %.2f[s]" % (self.time_interval*i)
        elems.append(ax.text(0.5, 9.5, time_str, fontsize=10))
        for obj in self.objects:
            obj.draw(ax, elems)
            if hasattr(obj, "one_step"): obj.one_step(self.time_interval)


# In[6]:


class IdealRobot:
    def __init__(self, pose, agent=None, sensor=None, color="black"):    # 引数を追加
        self.pose = pose
        self.r = 0.2
        self.color = color
        self.agent = agent
        self.poses = [pose]
        self.sensor = sensor    # 追加

    def draw(self, ax, elems):         ### call_agent_draw
        x, y, theta = self.pose
        xn = x + self.r * math.cos(theta)
        yn = y + self.r * math.sin(theta)
        elems += ax.plot([x,xn], [y,yn], color=self.color)
        c = patches.Circle(xy=(x, y), radius=self.r, fill=False, color=self.color)
        elems.append(ax.add_patch(c))
        self.poses.append(self.pose)
        elems += ax.plot([e[0] for e in self.poses], [e[1] for e in self.poses], linewidth=0.5, color="black")
        if self.sensor and len(self.poses) > 1:
            self.sensor.draw(ax, elems, self.poses[-2])
        if self.agent and hasattr(self.agent, "draw"):                               #以下2行追加
            self.agent.draw(ax, elems)

    @classmethod
    def state_transition(cls, nu, omega, time, pose):
        t0 = pose[2]
        if math.fabs(omega) < 1e-10:
            return pose + np.array( [nu*math.cos(t0),
                                     nu*math.sin(t0),
                                     omega ] ) * time
        else:
            return pose + np.array( [nu/omega*(math.sin(t0 + omega*time) - math.sin(t0)),
                                     nu/omega*(-math.cos(t0 + omega*time) + math.cos(t0)),
                                     omega*time ] )

    def one_step(self, time_interval):
        if not self.agent: return
        obs =self.sensor.data(self.pose) if self.sensor else None #追加
        nu, omega = self.agent.decision(obs) #引数追加
        self.pose = self.state_transition(nu, omega, time_interval, self.pose)
        if self.sensor: self.sensor.data(self.pose)


# In[7]:


class Agent:
    def __init__(self, nu, omega):
        self.nu = nu
        self.omega = omega

    def decision(self, observation=None):
        return self.nu, self.omega


# In[8]:


class Landmark:
    def __init__(self, x, y):
        self.pos = np.array([x, y]).T
        self.id = None

    def draw(self, ax, elems):
        c = ax.scatter(self.pos[0], self.pos[1], s=100, marker="*", label="landmarks", color="orange")
        elems.append(c)
        elems.append(ax.text(self.pos[0], self.pos[1], "id:" + str(self.id), fontsize=10))


# In[31]:


class Map:
    def __init__(self):       # 空のランドマークのリストを準備
        self.landmarks = []
        self.antennas = []

    def append_landmark(self, landmark):       # ランドマークを追加
        landmark.id = len(self.landmarks)           # 追加するランドマークにIDを与える
        self.landmarks.append(landmark)

    def append_antenna(self, antenna):
        antenna.id = len(self.antennas)
        self.antennas.append(antenna)

    def draw(self, ax, elems):                 # 描画（Landmarkのdrawを順に呼び出し）
        for ant in self.antennas: ant.draw(ax, elems)


# In[10]:


class IdealCamera:
    def __init__(self, env_map,                  distance_range=(0.5, 6.0),
                 direction_range=(-math.pi/3, math.pi/3)):
        self.map = env_map
        self.lastdata = []

        self.distance_range = distance_range
        self.direction_range = direction_range

    def visible(self, polarpos):  # ランドマークが計測できる条件
        if polarpos is None:
            return False

        return self.distance_range[0] <= polarpos[0] <= self.distance_range[1]                 and self.direction_range[0] <= polarpos[1] <= self.direction_range[1]

    def data(self, cam_pose):
        observed = []
        for lm in self.map.landmarks:
            z = self.observation_function(cam_pose, lm.pos)
            if self.visible(z):               # 条件を追加
                observed.append((z, lm.id))   # インデント

        self.lastdata = observed
        return observed

    @classmethod
    def observation_function(cls, cam_pose, obj_pos):
        diff = obj_pos - cam_pose[0:2]
        phi = math.atan2(diff[1], diff[0]) - cam_pose[2]
        while phi >= np.pi: phi -= 2*np.pi
        while phi < -np.pi: phi += 2*np.pi
        return np.array( [np.hypot(*diff), phi ] ).T

    def draw(self, ax, elems, cam_pose):
        for lm in self.lastdata:
            x, y, theta = cam_pose
            distance, direction = lm[0][0], lm[0][1]
            lx = x + distance * math.cos(direction + theta)
            ly = y + distance * math.sin(direction + theta)
            elems += ax.plot([x,lx], [y,ly], color="pink")


# In[36]:


class Antenna:
    def __init__(self, x, y):
        self.pos = np.array([x, y]).T
        self.id = None

    def draw(self, ax, elems):
        c = ax.scatter(self.pos[0], self.pos[1], s=100, marker="*", label="antennas", color="orange")
        elems.append(c)
        #elems.append(ax.text(self.pos[0], self.pos[1], "id:" + str(self.id), fontsize=10))


# In[29]:


class IdealAoA:
    def __init__(self, env_map,                  distance_range=(0.5, 6.0),
                 direction_range=(-math.pi/3, math.pi/3)):
        self.map = env_map
        self.lastdata = []

        self.distance_range = distance_range
        self.direction_range = direction_range

    def visible(self, theta):
        return True

    @classmethod
    def observation_function(cls, transmitter_pose, antenna_pos):
        diff = transmitter_pose[0:2] - antenna_pos
        theta = math.atan2(diff[1], diff[0])
        return np.array([0, theta]).T

    def data(self, transmitter_pose):
        observed = []
        for ant in self.map.antennas:
            z = self.observation_function(transmitter_pose, ant.pos)
            if self.visible(z):
                observed.append((z, ant.pos, ant.id))

        self.lastdata = observed

        return observed

    def draw(self, ax, elems, transmitter_pose):
        for ant in self.lastdata:
            z, pos, ant_id = ant
            theta = z[1]
            x = pos[0]
            y = pos[1]
            if theta > (math.pi/2):
                theta -= (math.pi/2)
                px = x - math.sin(theta) * 10
                py = y + math.cos(theta) * 10
            else:
                px = x + math.cos(theta) * 10
                py = y + math.sin(theta) * 10

            if ant_id == 0:
                elems += ax.plot([x,px], [y,py], color="pink")
                #elems.append(ax.text(0.5, 8, "id:" + str(ant_id) + " theta:" + str(theta*180/math.pi), fontsize=10))
            else:
                elems += ax.plot([x,px], [y,py], color="greenyellow")
                #elems.append(ax.text(0.5, 7.5, "id:" + str(ant_id) + " theta:" + str(theta*180/math.pi), fontsize=10))


# In[37]:


if __name__ == '__main__':   ###name_indent
    #world = World(30, 0.1, debug=True)
    world = World(30, 0.1)

    ### 地図を生成して3つランドマークを追加 ###
    m = Map()
    m.append_antenna(Antenna(0.5, 0.5))
    m.append_antenna(Antenna(9.5, 0.5))
    world.append(m)

    ### ロボットを作る ###
    straight = Agent(0.2, 0.0)
    circling = Agent(0.2, 10.0/180*math.pi)
    robot3 = IdealRobot( np.array([4, 6, math.pi/5*6]).T, sensor=IdealAoA(m), agent=circling, color="red")  # robot3は消しました
    world.append(robot3)

    ### アニメーション実行 ###
    world.draw()

