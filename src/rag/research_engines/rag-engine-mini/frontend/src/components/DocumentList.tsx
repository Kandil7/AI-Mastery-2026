'use client';

import React from 'react';
import { Document } from '@/types';
import { FileText, Trash2, Loader2, CheckCircle, AlertCircle, X } from 'lucide-react';

interface DocumentListProps {
  documents: Document[];
  onDelete: (documentId: string) => void;
  isLoading?: boolean;
}

const getFileIcon = (type: string) => {
  return <FileText className="w-8 h-8 text-blue-500" />;
};

const getStatusIcon = (status: Document['status']) => {
  switch (status) {
    case 'uploading':
      return <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />;
    case 'processing':
      return <Loader2 className="w-4 h-4 text-yellow-500 animate-spin" />;
    case 'indexed':
      return <CheckCircle className="w-4 h-4 text-green-500" />;
    case 'error':
      return <AlertCircle className="w-4 h-4 text-red-500" />;
    default:
      return null;
  }
};

const getStatusText = (status: Document['status']) => {
  switch (status) {
    case 'uploading':
      return 'Uploading...';
    case 'processing':
      return 'Processing...';
    case 'indexed':
      return 'Ready';
    case 'error':
      return 'Error';
    default:
      return '';
  }
};

const formatFileSize = (bytes: number) => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

export function DocumentList({ documents, onDelete, isLoading }: DocumentListProps) {
  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="w-8 h-8 text-blue-500 animate-spin" />
      </div>
    );
  }

  if (documents.length === 0) {
    return (
      <div className="text-center py-12 bg-gray-50 rounded-lg">
        <FileText className="w-12 h-12 text-gray-400 mx-auto mb-4" />
        <p className="text-gray-600 font-medium">No documents yet</p>
        <p className="text-gray-500 text-sm mt-1">
          Upload documents to start chatting with them
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {documents.map((doc) => (
        <div
          key={doc.id}
          className="flex items-center gap-4 p-4 bg-white border border-gray-200 rounded-lg hover:border-blue-300 transition-colors"
        >
          <div className="flex-shrink-0">
            {getFileIcon(doc.type)}
          </div>

          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <h4 className="font-medium text-gray-900 truncate">
                {doc.name}
              </h4>
              {doc.status === 'error' && doc.errorMessage && (
                <span className="text-xs text-red-600" title={doc.errorMessage}>
                  (hover for details)
                </span>
              )}
            </div>
            <div className="flex items-center gap-3 mt-1 text-sm text-gray-500">
              <span>{formatFileSize(doc.size)}</span>
              <span>•</span>
              <span className="uppercase">{doc.type}</span>
              <span>•</span>
              <span>{doc.createdAt.toLocaleDateString()}</span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5">
              {getStatusIcon(doc.status)}
              <span
                className={`text-sm font-medium ${
                  doc.status === 'error'
                    ? 'text-red-600'
                    : doc.status === 'indexed'
                    ? 'text-green-600'
                    : 'text-gray-600'
                }`}
              >
                {getStatusText(doc.status)}
              </span>
            </div>

            {doc.status === 'uploading' && doc.uploadProgress !== undefined && (
              <div className="w-24">
                <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-blue-500 transition-all duration-300"
                    style={{ width: `${doc.uploadProgress}%` }}
                  />
                </div>
                <span className="text-xs text-gray-500 mt-1">
                  {doc.uploadProgress}%
                </span>
              </div>
            )}

            {doc.status !== 'uploading' && doc.status !== 'processing' && (
              <button
                onClick={() => onDelete(doc.id)}
                className="p-2 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-colors"
                title="Delete document"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
