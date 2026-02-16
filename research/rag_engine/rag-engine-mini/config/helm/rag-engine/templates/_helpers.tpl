{{/*
Common labels for RAG Engine resources

التسمات المشتركة لموارد محرك RAG
*/}}
{{- range .Values.commonLabels }}
{{- with .Values.commonLabels }}
{{ .key }}: {{ .value | quote }}
{{- end }}
{{- end }}
