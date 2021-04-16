FROM maven:3.6.0-jdk-11-slim AS build
RUN apt-get update && \
apt-get install -y git


RUN git clone --single-branch --branch "es-7.5.0" https://github.com/lior-k/fast-elasticsearch-vector-scoring.git plugin
WORKDIR plugin
RUN mvn package


FROM docker.elastic.co/elasticsearch/elasticsearch:7.5.0
COPY --from=build /plugin/target/releases/elasticsearch-binary-vector-scoring-7.5.0.zip ./elasticsearch-binary-vector-scoring.zip
RUN ./bin/elasticsearch-plugin install file:///usr/share/elasticsearch/elasticsearch-binary-vector-scoring.zip