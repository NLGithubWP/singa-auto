import abc
import os
import time
import docker
import logging
from collections import namedtuple

from .container_manager import ContainerManager, ServiceRequirement, \
                                InvalidServiceRequestError, ContainerService

LABEL_AVAILBLE_GPUS = 'available_gpus'
LABEL_NUM_SERVICES = 'num_services'

logger = logging.getLogger(__name__)

_Node = namedtuple('_Node', ['id', 'available_gpus', 'num_services'])
_Deployment = namedtuple('_Deployment', ['node_id', 'gpu_nos'])

class DockerSwarmContainerManager(ContainerManager):
    def __init__(self,
        network=os.environ.get('DOCKER_NETWORK', 'rafiki')):
        self._network = network
        self._client = docker.from_env()

    def create_service(self, service_name, docker_image, replicas, 
                        args, environment_vars, mounts={}, publish_port=None,
                        requirements=[]) -> ContainerService:
        deployment = self._get_deployment(requirements)
        (service_id, hostname, port) \
            = self._create_service(deployment, service_name, docker_image, replicas, 
                                args, environment_vars, mounts, publish_port)
        info = {
            'node_id': deployment.node_id,
            'service_name': service_name,
            'replicas': replicas
        }
        service = ContainerService(service_id, hostname, port, info)
        self._mark_deployment(deployment)                            

        logger.info('Created service of ID "{}" with info {}'.format(service.id, service.info))
        return service

    def update_service(self, service: ContainerService, replicas):
        docker_service = self._client.services.get(service.id)
        docker_service.scale(replicas)

        logger.info('Updated service of ID "{}" to {} replicas' \
            .format(service.id, replicas))

    def destroy_service(self, service: ContainerService):
        self._destroy_sevice(service.id)
        node_id = service.info['node_id']
        deployment = _Deployment(node_id)
        self._unmark_deployment(deployment)
        logger.info('Deleted service of ID "{}"'.format(service.id))

    def _get_deployment(self, requirements) -> _Deployment:
        nodes = self._get_nodes()
        
        # Filter nodes with GPU if required
        if ServiceRequirement.GPU in requirements:
            nodes = [x for x in nodes if len(x.available_gpus) > 0]
        
        if len(nodes) == 0:
            raise InvalidServiceRequestError('There are no valid nodes to deploy the service on')
        
        # Choose the node with fewest services
        (_, node) = sorted([(x.num_services, x) for x in nodes])[0]

        deployment = _Deployment(node.id, node.available_gpus)
        return deployment

    def _mark_deployment(self, deployment):
        node_id = deployment.node_id

        # Update num services on node
        node = self._get_node(node_id)
        num_services = node.num_services + 1

        self._update_node(node_id, num_services)

    def _unmark_deployment(self, deployment):
        node_id = deployment.node_id
        
        # Update num services on node
        node = self._get_node(node_id)
        num_services = max(0, node.num_services - 1)

        self._update_node(node_id, num_services)

    def _destroy_sevice(self, service_id):
        service = self._client.services.get(service_id)
        service.remove()

    def _create_service(self, deployment, service_name, docker_image, replicas, 
                        args, environment_vars, mounts, publish_port):
        env = [
            '{}={}'.format(k, v)
            for (k, v) in environment_vars.items()
        ]
        mounts_list = [
            '{}:{}:rw'.format(k, v)
            for (k, v) in mounts.items()
        ]
        constraints = []

        ports_list = []
        container_port = None
        published_port = None
        hostname = service_name
        if publish_port is not None:
            # Host of Docker Swarm service = service's name at the container port
            published_port = int(publish_port[0])
            container_port = int(publish_port[1])
            ports_list = [{ 
                'PublishedPort': published_port, 
                'TargetPort': container_port
            }]

        # Modify service based on deployment info
        constraints.append('node.id=={}'.format(deployment.node_id)) # Add node constraint
        if deployment.use_gpu:
            env.append('CUDA_VISIBLE_DEVICES={}'.format(','.join(deployment.gpu_nos))) # GPU nos
        else:
            env.append('CUDA_VISIBLE_DEVICES=-1') # No GPU

        docker_service = self._client.services.create(
            image=docker_image,
            args=args,
            networks=[self._network],
            name=service_name,
            env=env,
            mounts=mounts_list,
            # Restart replicas when they exit with error
            restart_policy={
                'Condition': 'on-failure'
            },
            constraints=constraints,
            endpoint_spec={
                'Ports': ports_list
            },
            mode={
                'Replicated': {
                    'Replicas': replicas
                }
            }
        )

        # Host of Docker Swarm service = service's name at the container port
        return (docker_service.id, hostname, container_port)

    def _get_nodes(self):
        docker_nodes = self._client.nodes.list()
        nodes = [self._parse_node(x) for x in docker_nodes]
        return nodes

    def _get_node(self, node_id):
        docker_node = self._client.nodes.get(node_id)
        node = self._parse_node(docker_node)
        return node

    def _parse_node(self, docker_node):
        spec = docker_node.attrs.get('Spec', {})
        spec_labels = spec.get('Labels', {})
        available_gpus_str = spec_labels.get(LABEL_AVAILBLE_GPUS, '')
        available_gpus = [int(x) for x in available_gpus_str.split(',') if len(x) > 0]
        num_services = int(spec_labels.get(LABEL_NUM_SERVICES, 0))
        return _Node(docker_node.id, available_gpus, num_services)

    def _update_node(self, node_id, num_services):
        docker_node = self._client.nodes.get(node_id)
        spec = docker_node.attrs.get('Spec', {})
        spec_labels = spec.get('Labels', {})
        docker_node.update({
            **spec,
            'Labels': {
                **spec_labels,
                LABEL_NUM_SERVICES: str(num_services)
            }
        })
