FROM golang:1.25-alpine AS build
WORKDIR /src
COPY main.go .
RUN go mod init esproxy >/dev/null 2>&1 && CGO_ENABLED=0 go build -o /esproxy .

FROM scratch
COPY --from=build /esproxy /esproxy
ENTRYPOINT ["/esproxy"]
