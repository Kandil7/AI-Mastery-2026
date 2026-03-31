# Outputs for RAG Engine AWS Infrastructure

output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "cluster_name" {
  description = "Kubernetes Cluster Name"
  value       = module.eks.cluster_name
}

output "region" {
  description = "AWS region"
  value       = var.aws_region
}

output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "rag_engine_namespace" {
  description = "Kubernetes namespace for RAG Engine"
  value       = kubernetes_namespace.rag_engine.metadata[0].name
}

output "rag_engine_helm_release" {
  description = "Helm release name for RAG Engine"
  value       = helm_release.rag_engine.name
}

output "kubectl_config_command" {
  description = "Command to configure kubectl for the cluster"
  value       = "aws eks --region ${var.aws_region} update-kubeconfig --name ${module.eks.cluster_name}"
}