from manipulator_mujoco.robots.arm import Arm
import numpy as np

class SoArm(Arm):
    def __init__(self, xml_path, eef_site_name, attachment_site_name, joint_names = None, actuator_names = None, name: str = None):
        super(SoArm, self).__init__(xml_path, eef_site_name, attachment_site_name, joint_names, name)
        # Find MJCF elements that will be exposed as attributes.
        if actuator_names is None:
            self._actuator = self.mjcf_model.find_all('actuator')
        else:
            self._actuator = [self._mjcf_root.find('actuator', name) for name in actuator_names]
        

    @property
    def actuator(self):
        """List of actuator elements belonging to the arm."""
        return self._actuator
    
    def get_actuators(self, physics):
        all_actuators = physics.data.ctrl
        return all_actuators
    
    def set_actuators(self, physics, action):
        physics.bind(self.actuator).ctrl = action
        # setattr(actuators, "Jaw", action)
        # self._physics.bind(self._actuator)
        # physics.data.ctrl[0] = action
        # all_actuators = physics.data.ctrl
        # for i in range(len(all_actuators)):
        #     physics.data.ctrl[i] = action