#!/usr/bin/env python3
"""
Real-time visualization of attachment model dynamics.
"""
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import style
matplotlib.use('GTK3Agg')
import numpy as np
from matplotlib.widgets import Slider, Button
import rospy
from subscribers import ActionSubscriber
from Attachment-Model.msg import Emotion


class PlotGraph:
    """Real-time plotting of attachment behavior metrics."""
    
    def __init__(self):
        rospy.init_node('plot_graph')
        
        fig = plt.figure()
        fig.suptitle('MiRo Attachment Behavior', fontweight="bold", fontsize=18)

        # Emotional and physical distance
        self.dist_plot = fig.add_subplot(3, 2, 1)
        self.dist_plot.title.set_text("Distances")
        self.dist_plot.set_ylim(0, 1)

        # Child action
        self.child_action_plot = fig.add_subplot(3, 2, 3)
        self.child_action_plot.title.set_text("Child Action")
        self.child_action_plot.set_xlim(0, 1000)
        self.child_action_plot.set_yticks([0, 1], labels=('Approach', 'Explore'))
        self.child_action_plot.get_xaxis().set_visible(False)

        # Parent action
        self.parent_action_plot = fig.add_subplot(3, 2, 4)
        self.parent_action_plot.title.set_text("Parent Action")
        self.parent_action_plot.set_xlim(0, 1000)
        self.parent_action_plot.set_yticks([0, 1], labels=('Approach', 'Explore'))
        self.parent_action_plot.get_xaxis().set_visible(False)

        # Child need
        self.child_need_plot = fig.add_subplot(3, 2, 5)
        self.child_need_plot.title.set_text("Child Need")
        self.child_need_plot.set_xlim(0, 1000)
        self.child_need_plot.set_ylim(-2, 2)
        self.child_need_plot.get_xaxis().set_visible(False)

        # Parent need
        self.parent_need_plot = fig.add_subplot(3, 2, 6)
        self.parent_need_plot.title.set_text("Parent Need")
        self.parent_need_plot.set_xlim(0, 1000)
        self.parent_need_plot.set_ylim(-2, 2)
        self.parent_need_plot.get_xaxis().set_visible(False)

        # Data buffers
        self.child_actions = []
        self.parent_actions = []
        self.child_needs = []
        self.parent_needs = []

        # Sliders for attachment parameters
        plt.subplots_adjust(bottom=0.25)
        
        ax_av = plt.axes([0.25, 0.15, 0.5, 0.02])
        self.avoidant_slider = Slider(
            ax=ax_av, label='Avoidant', valmin=0, valmax=3, valinit=0
        )

        ax_am = plt.axes([0.25, 0.05, 0.5, 0.02])
        self.ambivalent_slider = Slider(
            ax=ax_am, label='Ambivalent', valmin=-0.1, valmax=2, valinit=0
        )

        # Reset button
        reset_ax = plt.axes([0.85, 0.04, 0.1, 0.04])
        self.reset_button = Button(reset_ax, 'Secure', hovercolor='0.975')
        self.reset_button.on_clicked(self._reset)

        # Publisher and subscriber
        self.emotion_pub = rospy.Publisher('/emotion', Emotion, queue_size=0)
        self.action_sub = ActionSubscriber()

    def _reset(self, event):
        """Reset sliders to secure attachment values."""
        self.avoidant_slider.reset()
        self.ambivalent_slider.reset()

    def _update_buffers(self):
        """Update data buffers with latest values."""
        self.child_actions.append(float(self.action_sub.child))
        if len(self.child_actions) > 1000:
            self.child_actions.pop(0)
            
        self.parent_actions.append(float(self.action_sub.parent))
        if len(self.parent_actions) > 1000:
            self.parent_actions.pop(0)
            
        self.child_needs.append(float(self.action_sub.child_need))
        if len(self.child_needs) > 1000:
            self.child_needs.pop(0)
            
        self.parent_needs.append(float(self.action_sub.parent_need))
        if len(self.parent_needs) > 1000:
            self.parent_needs.pop(0)

    def _update_plots(self):
        """Redraw all plots with current data."""
        # Distance bar chart
        distances = {
            "Emotional": self.action_sub.emotional_distance,
            "Physical": self.action_sub.physical_distance
        }
        self.dist_plot.clear()
        self.dist_plot.title.set_text("Distances")
        self.dist_plot.set_ylim(0, 1)
        bars = self.dist_plot.bar(distances.keys(), distances.values(), width=0.4)
        bars[0].set_color('maroon')
        bars[1].set_color('blue')

        # Child action
        self.child_action_plot.clear()
        self.child_action_plot.set_xlim(0, 1000)
        self.child_action_plot.set_ylim(-0.2, 1.2)
        self.child_action_plot.title.set_text("Child Action")
        self.child_action_plot.plot(self.child_actions, color="red", linewidth=2)
        self.child_action_plot.set_yticks([0, 1], labels=('Approach', 'Explore'))
        self.child_action_plot.get_xaxis().set_visible(False)

        # Parent action
        self.parent_action_plot.clear()
        self.parent_action_plot.set_xlim(0, 1000)
        self.parent_action_plot.set_ylim(-0.2, 1.2)
        self.parent_action_plot.title.set_text("Parent Action")
        self.parent_action_plot.plot(self.parent_actions, color="brown", linewidth=2)
        self.parent_action_plot.set_yticks([0, 1], labels=('Approach', 'Explore'))
        self.parent_action_plot.get_xaxis().set_visible(False)

        # Child need
        self.child_need_plot.clear()
        self.child_need_plot.set_xlim(0, 1000)
        self.child_need_plot.set_ylim(-2, 2)
        self.child_need_plot.title.set_text("Child Need")
        self.child_need_plot.plot(self.child_needs, color="red", linewidth=2)
        self.child_need_plot.get_xaxis().set_visible(False)

        # Parent need
        self.parent_need_plot.clear()
        self.parent_need_plot.set_xlim(0, 1000)
        self.parent_need_plot.set_ylim(-2, 2)
        self.parent_need_plot.title.set_text("Parent Need")
        self.parent_need_plot.plot(self.parent_needs, color="brown", linewidth=2)
        self.parent_need_plot.get_xaxis().set_visible(False)

    def run(self):
        """Main animation loop."""
        rate = rospy.Rate(10)
        
        while not rospy.is_shutdown():
            self._update_buffers()
            self._update_plots()
            
            # Publish emotion parameters
            emotion_msg = Emotion()
            emotion_msg.avoidant = self.avoidant_slider.val
            emotion_msg.ambivalent = self.ambivalent_slider.val
            self.emotion_pub.publish(emotion_msg)
            
            plt.pause(0.05)
            rate.sleep()


if __name__ == '__main__':
    try:
        plotter = PlotGraph()
        plotter.run()
    except rospy.ROSInterruptException:
        pass

        pass