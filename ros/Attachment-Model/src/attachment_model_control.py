#!/usr/bin/env python3
"""
Central controller - attachment model dynamics and action selection.
"""
import sys
sys.path.insert(0, '/workspace/.venv/lib/python3.8/site-packages')
import rospy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import threading
from attachment_model.msg import Action
from subscribers import EmotionSubscriber

# Neural Network Models
class PhysicalDistanceNN(nn.Module):
    def __init__(self, D_in):
        super(PhysicalDistanceNN, self).__init__()
        hidden_dim = 32
        self.fc1 = nn.Linear(D_in, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2*hidden_dim)
        self.fc3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc4(x)
        return x

class EmotionalDistanceNN(nn.Module):
    def __init__(self, D_in):
        super(EmotionalDistanceNN, self).__init__()
        hidden_dim = 64
        self.fc1 = nn.Linear(D_in, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2*hidden_dim)
        self.fc3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        x = F.silu(self.fc3(x))
        x = self.fc4(x)
        return x

class CentralAttachmentController:
    """Central controller implementing attachment dynamics."""
    
    def __init__(self):
        rospy.init_node(f'central_control')

        # Publishers
        self.controller_publisher = rospy.Publisher(f'/central/controller', Action, queue_size=0)
        
        # Subscribers
        self.emotion = EmotionSubscriber()
        
        # Set variables for the node
        self.time = 0
        self.child_need = -1.0
        self.child_accumulated_need = 1.0
        self.parent_need = -1.0
        self.parent_accumulated_need = 1.0
        self.k0 = 0.8
        self.a = 1.0
        self.c = 1.0
        self.b = 0.7
        self.delta = 0.1
        self.physical_distance = 0.0
        self.emotional_distance = 0.0
        self.epsilonAv = self.emotion.avoidant
        self.epsilonAm = self.emotion.ambivalent
        self.h = 0.01
        
        # Action message initialization
        self.action_msg = Action()
        self.action_msg.child = 0
        self.action_msg.parent = 0
        self.action_msg.emotional_distance = 0
        self.action_msg.physical_distance = 0
        self.action_msg.child_need = 0
        self.action_msg.parent_need = 0
        self.action_msg.child_accumulated_need = 0
        self.action_msg.parent_accumulated_need = 0

    def _calculate_physical_distance(self):
        """Calculate physical distance based on model"""
        return (self.child_accumulated_need + self.parent_accumulated_need)/2 + self.child_accumulated_need

    def _calculate_emotional_distance(self):
        """Calculate emotional distance based on model"""
        return self.k0*((self.child_accumulated_need - self.parent_accumulated_need)/2 + self.child_accumulated_need) + self.delta*self.k0

    def run(self):
        """Main control loop implementing attachment dynamics."""
        rate = rospy.Rate(30)
        A_explore = lambda x: np.heaviside(-x,0)
        
        while not rospy.is_shutdown():
            self.time += self.h

            # Calculate physical distance (normalized to [0,1])
            self.physical_distance = self._calculate_physical_distance()
            
            # Calculate emotional distance
            self.emotional_distance = self._calculate_emotional_distance()
            
            calculated_need_accumulation = [self.child_need, self.child_accumulated_need, self.parent_need, self.parent_accumulated_need]

            # Calculate needs and accumulated needs, Runga-Kutta 4th order
            k1 = self.f(self.time, calculated_need_accumulation)
            k2 = self.f(self.time + self.h/2.0, calculated_need_accumulation + self.h*k1/2.0)
            k3 = self.f(self.time + self.h/2.0, calculated_need_accumulation + self.h*k2/2.0)
            k4 = self.f(self.time + self.h, calculated_need_accumulation + self.h*k3)
            calculated_need_accumulation = calculated_need_accumulation + self.h*(k1 + 2*k2 + 2*k3 + k4)/6.0

            # Update action message
            self.action_msg.child = int(A_explore(calculated_need_accumulation[0]))
            self.action_msg.parent = int(A_explore(calculated_need_accumulation[2]))
            self.action_msg.physical_distance = self.physical_distance
            self.action_msg.emotional_distance = self.emotional_distance
            self.action_msg.child_need = calculated_need_accumulation[0]
            self.action_msg.parent_need = calculated_need_accumulation[2]
            self.action_msg.child_accumulated_need = calculated_need_accumulation[1]
            self.action_msg.parent_accumulated_need = calculated_need_accumulation[3]
            self.controller_publisher.publish(self.action_msg)

            # Update model variables
            [self.child_need,self.child_accumulated_need,self.parent_need,self.parent_accumulated_need] = calculated_need_accumulation
            rate.sleep()

    # Model
    def f(self, t, r):
        x1 = r[0]
        y1 = r[1]
        x2 = r[2]
        y2 = r[3]
        ym = (y1 + y2)/2.0
        xm = (x1 + x2)/2.0
        # Calculate needs and accumulated needs
        dx1 = -self.a*(4.0*self.c*x1**3 - 2.0*x1) - y1
        dy1 = self.b*x1- self.epsilonAm*(y2 + y1 - self.physical_distance)  - self.epsilonAv*(y2 - y1 - self.emotional_distance)
        dx2 = -self.a*(4.0*self.c*x2**3 - 2.0*x2) - y2
        dy2 = self.b*x2  - self.epsilonAm*(y2 + y1 + self.physical_distance)  - self.epsilonAv*(y2 - y1 - self.emotional_distance)
        return np.array([dx1, dy1, dx2, dy2])

class IndividualAttachmentController:
    """Individual controller implementing attachment dynamics."""
    
    def __init__(self, robot_name):
        # Don't initialize node here - will be done in main
        
        # Publishers
        self.controller_publisher = rospy.Publisher(f'/{robot_name}/controller', Action, queue_size=10)
        
        # Subscribers
        self.emotion = EmotionSubscriber()
        
        # Store robot name
        self.robot_name = robot_name
        
        # Set variables for the node
        self.time = 0
        self.child_need = -1.0
        self.child_accumulated_need = 1.0
        self.parent_need = -1.0
        self.parent_accumulated_need = 1.0
        self.k0 = 0.8
        self.a = 1.0
        self.c = 1.0
        self.b = 0.7
        self.delta = 0.1
        self.physical_distance = 0.0
        self.emotional_distance = 0.0
        self.epsilonAv = self.emotion.avoidant
        self.epsilonAm = self.emotion.ambivalent
        self.h = 0.01
        
        # Action message initialization
        self.action_msg = Action()
        self.action_msg.child = 0
        self.action_msg.parent = 0
        self.action_msg.emotional_distance = 0
        self.action_msg.physical_distance = 0
        self.action_msg.child_need = 0
        self.action_msg.parent_need = 0
        self.action_msg.child_accumulated_need = 0
        self.action_msg.parent_accumulated_need = 0
        
        # Load neural network models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dir = os.path.join(os.path.dirname(__file__))
        
        # Load coupling models
        self.physical_NN = PhysicalDistanceNN(D_in=4).to(self.device)
        self.emotional_NN = EmotionalDistanceNN(D_in=6).to(self.device)
        
        try:
            self.physical_NN.load_state_dict(
                torch.load(os.path.join(model_dir, 'physical_parent_coupling_model.pth'), 
                          map_location=self.device))
            self.emotional_NN.load_state_dict(
                torch.load(os.path.join(model_dir, 'emotional_parent_coupling_model.pth'), 
                          map_location=self.device))
            
            self.physical_NN.eval()
            self.emotional_NN.eval()
            
            rospy.loginfo("Successfully loaded neural network models")
        except Exception as e:
            rospy.logwarn(f"Failed to load neural network models: {e}. Using fallback calculations.")
            self.physical_NN = None
            self.emotional_NN = None

    def _calculate_physical_distance(self):
        """Calculate physical distance using neural network or fallback to model"""
        if self.physical_NN is not None:
            try:
                # Input: [distance, x_state, y_state, ambivalence]
                # Use euclidean distance between accumulated needs
                distance = np.sqrt((self.child_accumulated_need - self.parent_accumulated_need)**2)
                if self.robot_name == "child":
                    need = self.child_need
                    accumulated_need = self.child_accumulated_need
                else:
                    need = self.parent_need
                    accumulated_need = self.parent_accumulated_need
                input_tensor = torch.tensor(
                    [distance, 
                     need, 
                     accumulated_need, 
                     self.epsilonAm],
                    dtype=torch.float32
                ).to(self.device)
                
                with torch.no_grad():
                    coupling = self.physical_NN(input_tensor).item()
                
                return coupling
            except Exception as e:
                rospy.logwarn_throttle(10, f"Error in physical NN: {e}. Using fallback.")
        
        # Fallback to original calculation
        return (self.child_accumulated_need + self.parent_accumulated_need)/2 + self.child_accumulated_need

    def _calculate_emotional_distance(self):
        """Calculate emotional distance using neural network or fallback to model"""
        if self.emotional_NN is not None:
            try:
                # Input: [evidence_trace, other_vocal, self_vocal, x_state, y_state, avoidance]
                # For simplicity, use zeros for vocalization tracking
                if self.robot_name == "child":
                    need = self.child_need
                    accumulated_need = self.child_accumulated_need
                else:
                    need = self.parent_need
                    accumulated_need = self.parent_accumulated_need
                input_tensor = torch.tensor(
                    [0.0,  # evidence_trace (could track this over time if needed)
                     0,    # other vocalization
                     0,    # self vocalization
                     need,
                     accumulated_need,
                     self.epsilonAv],
                    dtype=torch.float32
                ).to(self.device)
                
                with torch.no_grad():
                    coupling = self.emotional_NN(input_tensor).item()
                
                return coupling
            except Exception as e:
                rospy.logwarn_throttle(10, f"Error in emotional NN: {e}. Using fallback.")
        
        # Fallback to original calculation
        return self.k0*((self.child_accumulated_need - self.parent_accumulated_need)/2 + self.child_accumulated_need) + self.delta*self.k0

    def run(self):
        """Main control loop implementing attachment dynamics."""
        rate = rospy.Rate(30)
        A_explore = lambda x: np.heaviside(-x,0)
        
        while not rospy.is_shutdown():
            self.time += self.h

            # Calculate physical distance (normalized to [0,1])
            self.physical_distance = self._calculate_physical_distance()
            
            # Calculate emotional distance
            self.emotional_distance = self._calculate_emotional_distance()
            
            calculated_need_accumulation = [self.child_need, self.child_accumulated_need, self.parent_need, self.parent_accumulated_need]

            # Calculate needs and accumulated needs, Runga-Kutta 4th order
            k1 = self.f(self.time, calculated_need_accumulation)
            k2 = self.f(self.time + self.h/2.0, calculated_need_accumulation + self.h*k1/2.0)
            k3 = self.f(self.time + self.h/2.0, calculated_need_accumulation + self.h*k2/2.0)
            k4 = self.f(self.time + self.h, calculated_need_accumulation + self.h*k3)
            calculated_need_accumulation = calculated_need_accumulation + self.h*(k1 + 2*k2 + 2*k3 + k4)/6.0

            # Update action message
            self.action_msg.child = int(A_explore(calculated_need_accumulation[0]))
            self.action_msg.parent = int(A_explore(calculated_need_accumulation[2]))
            self.action_msg.physical_distance = self.physical_distance
            self.action_msg.emotional_distance = self.emotional_distance
            self.action_msg.child_need = calculated_need_accumulation[0]
            self.action_msg.parent_need = calculated_need_accumulation[2]
            self.action_msg.child_accumulated_need = calculated_need_accumulation[1]
            self.action_msg.parent_accumulated_need = calculated_need_accumulation[3]
            self.controller_publisher.publish(self.action_msg)

            # Update model variables
            [self.child_need,self.child_accumulated_need,self.parent_need,self.parent_accumulated_need] = calculated_need_accumulation
            rate.sleep()

    # Model
    def f(self, t, r):
        x1 = r[0]
        y1 = r[1]
        x2 = r[2]
        y2 = r[3]
        ym = (y1 + y2)/2.0
        xm = (x1 + x2)/2.0
        # Calculate needs and accumulated needs
        dx1 = -self.a*(4.0*self.c*x1**3 - 2.0*x1) - y1
        dy1 = self.b*x1- self.epsilonAm*(y2 + y1 - self.physical_distance)  - self.epsilonAv*(y2 - y1 - self.emotional_distance)
        dx2 = -self.a*(4.0*self.c*x2**3 - 2.0*x2) - y2
        dy2 = self.b*x2  - self.epsilonAm*(y2 + y1 + self.physical_distance)  - self.epsilonAv*(y2 - y1 - self.emotional_distance)
        return np.array([dx1, dy1, dx2, dy2])

if __name__ == '__main__':
    model_separation = True
    try:
        if model_separation is False:
            controller = CentralAttachmentController()
            controller.run()
        else:
            # Initialize node once for both controllers
            rospy.init_node('individual_attachment_controllers')
            
            child_controller = IndividualAttachmentController(robot_name="child")
            parent_controller = IndividualAttachmentController(robot_name="parent")
            # Run both controllers in parallel using threads
            child_thread = threading.Thread(target=child_controller.run)
            parent_thread = threading.Thread(target=parent_controller.run)
            child_thread.start()
            parent_thread.start()
            child_thread.join()
            parent_thread.join()
    except rospy.ROSInterruptException:
        pass
