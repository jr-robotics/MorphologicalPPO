from typing import Callable, Dict, Union, Optional
from collections import OrderedDict
import importlib.resources
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces
from gymnasium.envs.mujoco.swimmer_v5 import SwimmerEnv
from src.envs.spaces import AugmentedKinematicGraph
from src.envs.spaces import KinematicGraphInstance
from numpy.typing import NDArray

import hashlib

from src.envs.common.parameter import Parameter



def custom_pretty_print(elem, level=0):
    """Recursively prettify XML elements."""
    indent = '  ' * level
    if len(elem) > 0:
        # Only indent the opening tag
        pretty_string = f"{indent}<{elem.tag}"
        for name, value in elem.attrib.items():
            pretty_string += f' {name}="{value}"'
        pretty_string += '>\n'
        
        # Recursively prettify children
        for child in elem:
            pretty_string += custom_pretty_print(child, level + 1)
        pretty_string += f"{indent}</{elem.tag}>\n"
    else:
        pretty_string = f"{indent}<{elem.tag}"
        for name, value in elem.attrib.items():
            pretty_string += f' {name}="{value}"'
        pretty_string += '/>\n'
    return pretty_string

def gen_swimmer_config(num_seg=3, len_seg=1, radius=0.1, density=1000, gear=150):
    # Load the header from the original swimmer.xml
    with importlib.resources.open_text('gymnasium.envs.mujoco.assets', 'swimmer.xml') as f:
        header = f.read()

    # Parse the header
    root = ET.fromstring(header)

    # Find the actuator element and clear existing actuators
    actuator = root.find('actuator')
    actuator.clear()

    # Remove the original swimmer body
    worldbody = root.find('worldbody')

    # Find and keep the torso body element as is
    torso = worldbody.find("./body[@name='torso']")

    # Set new position for the torso coordnate system. We want to have at the which 
    # is exactly at num_seg*len_seg/2
    torso.set("pos", f"{num_seg*len_seg/2} 0 0") 
    torso_geom = torso.find("geom")
    torso_geom.set("fromto", f"0 0 0 {-len_seg} 0 0")
    torso_geom.set("size", str(radius))

    # Alter camea position to new setup.
    camera = torso.find("camera")
    camera.set("pos", f"{-num_seg*len_seg/2} {-num_seg*len_seg} {num_seg*len_seg}")  # Position the camera back and above
   

    # Remove all other bodies under torso (which would be segments)
    for body in torso.findall('body'):
        torso.remove(body)

    parent = torso

    for i in range(1, num_seg):

        # Create a new body for the segment
        new_body = ET.SubElement(
            parent,
            "body",
            name="segment_%d"%i,
            pos=f"{-len_seg} 0 0",
        )
        
        # Add a geometry tag to the body
        ET.SubElement(
            new_body,
            "geom",
            density=str(density),
            fromto=f"0 0 0 {-len_seg} 0 0",
            size=f"{radius}",
            type="capsule",
        )

        # Add a joint tag to the body
        joint_name = "motor_%d_rot"%i
        ET.SubElement(
            new_body, 
            "joint",
            name=joint_name,
            axis="0 0 1",
            limited="true",
            pos="0 0 0",
            range="-100 100",
            type="hinge"
        )    
    
        ET.SubElement(
            actuator,
            "motor",
            ctrllimited="true",
            ctrlrange="-1 1",
            gear=f"{gear}",
            joint=joint_name,
        )

        parent = new_body


    return custom_pretty_print(root)



        
class GraphSwimmerEnv(SwimmerEnv):
    
    def __init__(
        self,
        xml_file: str = "swimmer.xml",
        frame_skip: int = 4,
        default_camera_config: Dict[str, Union[float, int]] = {},
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 1e-4,
        reset_noise_scale: float = 0.1,
        exclude_current_positions_from_observation: bool = True,
        use_node_attr: bool = True,
        return_dummy_node_attr: bool = True,
        stack_edge_attr: bool = True,
        dtype: Union[np.float32 | np.float64] = np.float32,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            ctrl_cost_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            use_node_attr,
            return_dummy_node_attr,
            stack_edge_attr,
            **kwargs,
        )

        self._frame_skip = frame_skip
        self._default_camera_config = default_camera_config
        self._return_dummy_node_attr = return_dummy_node_attr
        self._stack_edge_attr = stack_edge_attr
        self._dtype = dtype
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        
        self._kwargs = kwargs
        
        self._setup(xml_file=xml_file)


        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }
    
    
    def _setup(self, xml_file:str ):
        MujocoEnv.__init__(
            self,
            xml_file,
            self._frame_skip,
            observation_space=None,
            default_camera_config=self._default_camera_config,
            **self._kwargs,
        )
      
        
        # parse xml config and get link and joint parameters
        self.parse_config(xml=xml_file)
        
        
        
        
        if self._return_dummy_node_attr:
            # Create dummy node attr
            self.node_attr = np.ones(len(self.links.values())).reshape(-1,1)
        else:   
            # Extract node features from links
            self.node_attr = []
            for link in self.links.values():
                link_params = []
                fromto = [float(v) for v in link["fromto"].split()]
                link_params.append(float(link["density"]))
                link_params.append(float(link["size"]))
                link_params.append(np.linalg.norm(np.array(fromto[:3]) - np.array(fromto[3:])))
                self.node_attr.append(link_params)

            self.node_attr = np.asarray(self.node_attr)

        # Extract edge features from joints
        self.edge_attr = []
        for joint in self.joints.values():
            joint_params = []
            joint_params.append(float(joint["gear"]))
            self.edge_attr.append(joint_params)
        self.edge_attr = np.asarray(self.edge_attr)

        edge_indices = [(i, i+1) for i in range(0, len(self.links)-1)]

        self.observation_space = AugmentedKinematicGraph(
            global_space=spaces.Box(low=-np.inf, high=np.inf, shape=(self.global_obs_size,), dtype=self._dtype),
            node_attr_shape=(self.node_attr_size,),
            edge_attr_shape=(self.edge_attr_size,),
            edge_indices=edge_indices,
            dtype=self._dtype,
        )
    
    
    @property
    def global_obs_size(self):
        return len(self.global_joints)*2 - 2 if \
            self._exclude_current_positions_from_observation else len(self.global_joints)*2
            
    @property
    def edge_attr_size(self):
        return 2 + self.edge_attr.shape[-1] if self._stack_edge_attr else 2
    
    @property
    def node_attr_size(self):
        return self.node_attr.shape[1]
    

    def parse_config(self, xml):
        with open(xml, "r") as f:
            config = ET.fromstring(f.read())
        worldbody = config.find("worldbody")
        bodies = worldbody.findall('.//body')

        
        self.links = OrderedDict()
        self.global_joints = OrderedDict()
        self.joints = OrderedDict()
        
        
        for body in bodies:
            geometry = body.find("geom")
            self.links[body.get("name")] = {k:geometry.get(k) for k in geometry.keys()}
            
            
            if body.get("name") == "torso":
                for joint in body.findall("joint"):
                    self.global_joints[joint.get("name")] = {k:joint.get(k) for k in joint.keys() if k != "name"}
                    
            else:
                joint = body.find("joint")
                self.joints[joint.get("name")] = {k:joint.get(k) for k in joint.keys() if k != "name"}
                
                
        # append motor config to joint dictionaries        
        for motor in config.findall(".//motor"):
            self.joints[motor.get("joint")].update({k:motor.get(k) for k in motor.keys() if k != "joint"})


    def _get_edge_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()
        
        
        edge_obs = np.zeros(
            shape=self.observation_space.shape["edge_space"],
            dtype=self.observation_space.edge_space.dtype,
        )
        
        edge_obs[:,0] = position[len(self.global_joints):]
        edge_obs[:,1] = velocity[len(self.global_joints):]        
        
        if self._stack_edge_attr:
            edge_obs[:,2:] = self.edge_attr
        
        return edge_obs


    def _get_obs(self) -> Dict[str, Union[NDArray, KinematicGraphInstance]]:
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()

        global_position = position[:len(self.global_joints)]
        global_velocity = velocity[:len(self.global_joints)]
        
        if self._exclude_current_positions_from_observation:
            global_position = global_position[2:]
            
        global_obs = np.concatenate([global_position, global_velocity]).ravel()
        
    
        return dict(
            global_space = global_obs.astype(self._dtype),
            node_space = self.node_attr.astype(self._dtype),
            edge_space = self._get_edge_obs().astype(self._dtype),
            edge_indices = self.observation_space.edge_indices
        )

    

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            #"edge_indices": self.observation_space.edge_indices,
        }
        

      
        
class ParameterizebleSwimmerEnv(GraphSwimmerEnv):
    def __init__(
        self,
        xml_params: dict,
        xml_path: str,
        xml_creator: Callable = gen_swimmer_config,
        seed: Optional[int] = ModuleNotFoundError,
        **kwargs,
        ):
        
        self._xml_creator = xml_creator
        self._xml_path = Path(xml_path)
        self._xml_params = xml_params
        self._seed = seed
        self._kwargs = kwargs
        
        self._xml_path.mkdir(parents=True, exist_ok=True)
        
        if self._seed is not None:
            for key in self._xml_params.keys():
                if isinstance(self._xml_params[key], spaces.Space):
                    self._xml_params[key].seed(self._seed)
            
        
        self._sampled_xml_params = self._sample_xml_params()
        
        
        # xml_str = xml_creator(**xml_params)
        # sha_256 = hashlib.sha256()
        # sha_256.update(xml_str.encode())
        
        # xml_path = Path(xml_path)
        # xml_path.mkdir(parents=True, exist_ok=True)
        # xml_path = xml_path.joinpath(sha_256.hexdigest() + ".xml")
        
        # with open(xml_path.as_posix(), "w") as fid:
        #     fid.write(xml_str)
            
        super().__init__(
            xml_file=self.xml_file,
            **kwargs,
            )

    @property
    def xml_file(self):
        xml_str = self._xml_creator(**self._sampled_xml_params)
        sha_256 = hashlib.sha256()
        sha_256.update(xml_str.encode())
        
        xml_file = self._xml_path.joinpath(sha_256.hexdigest() + ".xml").as_posix()
                
        with open(xml_file, "w") as fid:
            fid.write(xml_str)
        
        return xml_file
            

    def _sample_xml_params(self):
        sampled_xml_params = dict()
        
        for key, param in self._xml_params.items():
            sampled_xml_params[key] = param() if isinstance(param, Parameter) else param
            
        return sampled_xml_params

    
            
    def reset(self, *, seed = None, options = None):
            
        # sample new xml parameters or take values as defined in the xml_params
        old_sampled_params = self._sampled_xml_params
        self._sampled_xml_params = self._sample_xml_params()
           
        if self._sampled_xml_params != old_sampled_params:
            super()._setup(xml_file=self.xml_file)

        return super().reset(seed=seed, options=options)
        
    
    
    
    
        
        
  
        
      
class FlattenGraphEnv(ParameterizebleSwimmerEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        obs_size = (
            self.data.qpos.size
            + self.data.qvel.size
            - 2 * self._exclude_current_positions_from_observation
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "skipped_qpos": 2 * self._exclude_current_positions_from_observation,
            "qpos": self.data.qpos.size
            - 2 * self._exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size,
        }

    def _get_obs(self):
        return SwimmerEnv._get_obs(self)
    
    
    
    
# class ResetSampledSwimmerEnv(ParameterizebleSwimmerEnv):
#     def __init__(
#         self,
#         xml_params,
#         xml_path,
#         xml_creator = gen_swimmer_config,
#         **kwargs
#         ):
#         self._xml_params = xml_params
#         self._xml_path = xml_path
#         self._xml_creator = xml_creator
#         self._kwargs = kwargs
                    
#         self._sampled_xml_params = self._sample_xml_params()
                    
#         super().__init__(
#             xml_params=self._sample_xml_params(), 
#             xml_path=xml_path, 
#             xml_creator=xml_creator, 
#             **kwargs,
#         )
        

#     def _sample_xml_params(self):
#         sampled_xml_params = dict()
        
#         for key, param in self._xml_params.items():
#             sampled_xml_params[key] = param() if isinstance(param, Parameter) else param
            
#         return sampled_xml_params
    
    

    
#     def reset(self, *, seed = None, options = None):
        
#         old_sampled_params = self._sampled_xml_params
#         self._sampled_xml_params = self._sample_xml_params()
        
#         if self._sampled_xml_params != old_sampled_params:
#             super().__init__(
#                 xml_params=self._sampled_xml_params, 
#                 xml_path=self._xml_path, 
#                 xml_creator=self._xml_creator, 
#                 **self._kwargs,
#             )

#         return super().reset(seed=seed, options=options)
