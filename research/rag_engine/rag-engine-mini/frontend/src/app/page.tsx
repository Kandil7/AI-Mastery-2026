'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { ChatWindow } from '@/components/ChatWindow';
import { DocumentList } from '@/components/DocumentList';
import { UploadModal } from '@/components/UploadModal';
import { Button } from '@/components/ui/Button';
import { PlusCircle, FileText, MessageSquare, AlertCircle, Loader2 } from 'lucide-react';
import { Document } from '@/types';
import { getDocuments, deleteDocument, ApiError } from '@/lib/api';

export default function Home() {
  const [activeTab, setActiveTab] = useState<'chat' | 'documents'>('chat');
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchDocuments = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      const docs = await getDocuments();
      setDocuments(docs);
    } catch (err) {
      const message = err instanceof ApiError ? err.message : 'Failed to load documents';
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  const handleDelete = async (documentId: string) => {
    try {
      await deleteDocument(documentId);
      setDocuments((prev) => prev.filter((d) => d.id !== documentId));
    } catch (err) {
      const message = err instanceof ApiError ? err.message : 'Failed to delete document';
      alert(message);
    }
  };

  return (
    <main className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-sm">RAG</span>
            </div>
            <h1 className="text-xl font-semibold text-gray-900">RAG Engine</h1>
          </div>
          
          <Button 
            onClick={() => setIsUploadOpen(true)}
            className="flex items-center space-x-2"
          >
            <PlusCircle className="w-4 h-4" />
            <span>Upload Document</span>
          </Button>
        </div>
      </header>

      {/* Navigation Tabs */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-6">
        <div className="flex space-x-1 bg-gray-100 p-1 rounded-lg w-fit">
          <button
            onClick={() => setActiveTab('chat')}
            className={`flex items-center space-x-2 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              activeTab === 'chat'
                ? 'bg-white text-gray-900 shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            <MessageSquare className="w-4 h-4" />
            <span>Chat</span>
          </button>
          <button
            onClick={() => setActiveTab('documents')}
            className={`flex items-center space-x-2 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              activeTab === 'documents'
                ? 'bg-white text-gray-900 shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            <FileText className="w-4 h-4" />
            <span>Documents</span>
            {documents.length > 0 && (
              <span className="ml-1 px-2 py-0.5 bg-blue-100 text-blue-700 text-xs font-medium rounded-full">
                {documents.length}
              </span>
            )}
          </button>
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-4">
          <div className="p-4 bg-red-50 border border-red-200 rounded-lg flex items-center gap-3">
            <AlertCircle className="w-5 h-5 text-red-500" />
            <span className="text-red-700">{error}</span>
            <button
              onClick={fetchDocuments}
              className="ml-auto px-3 py-1 bg-red-100 hover:bg-red-200 text-red-700 text-sm rounded-md transition-colors"
            >
              Retry
            </button>
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {activeTab === 'chat' ? (
          <ChatWindow />
        ) : (
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
            <div className="px-6 py-4 border-b border-gray-200">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <FileText className="w-5 h-5 text-gray-600" />
                  <h2 className="font-semibold text-gray-900">Your Documents</h2>
                  {documents.length > 0 && (
                    <span className="px-2 py-0.5 bg-gray-100 text-gray-600 text-xs font-medium rounded-full">
                      {documents.length}
                    </span>
                  )}
                </div>
                <button
                  onClick={() => setIsUploadOpen(true)}
                  className="p-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors"
                >
                  <PlusCircle className="w-4 h-4" />
                </button>
              </div>
            </div>
            <div className="p-6">
              {isLoading ? (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="w-8 h-8 text-blue-500 animate-spin" />
                </div>
              ) : (
                <DocumentList
                  documents={documents}
                  onDelete={handleDelete}
                  isLoading={false}
                />
              )}
            </div>
          </div>
        )}
      </div>

      {/* Upload Modal */}
      <UploadModal 
        isOpen={isUploadOpen} 
        onClose={() => setIsUploadOpen(false)}
        onUploadComplete={() => {
          fetchDocuments();
          setIsUploadOpen(false);
        }}
      />
    </main>
  );
}