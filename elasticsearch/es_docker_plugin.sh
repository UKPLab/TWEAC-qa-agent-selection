docker run -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" -e "cluster.routing.allocation.disk.threshold_enabled=false" --name "YOUR_NAME" --mount 'type=bind,source=PATH_TO_INDEX' es-vector:latest

