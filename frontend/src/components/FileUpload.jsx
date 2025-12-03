import React, { useState } from 'react';
import axios from 'axios';

const FileUpload = ({ onUploadSuccess }) => {
    const [file, setFile] = useState(null);
    const [uploading, setUploading] = useState(false);
    const [error, setError] = useState(null);
    const [isDragging, setIsDragging] = useState(false);

    const handleFileChange = (e) => {
        setFile(e.target.files[0]);
        setError(null);
    };

    const handleDragOver = (e) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = () => {
        setIsDragging(false);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragging(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            setFile(e.dataTransfer.files[0]);
            setError(null);
        }
    };

    const handleUpload = async () => {
        if (!file) {
            setError("Please select a file first.");
            return;
        }

        setUploading(true);
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await axios.post('http://localhost:5000/api/upload', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            onUploadSuccess(response.data);
        } catch (err) {
            console.error(err);
            setError("Upload failed. Please try again.");
        } finally {
            setUploading(false);
        }
    };

    return (
        <div className="relative group">
            <div className="absolute -inset-1 bg-gradient-to-r from-pink-400 to-purple-600 rounded-2xl blur opacity-25 group-hover:opacity-50 transition duration-1000 group-hover:duration-200"></div>
            <div className="relative bg-white/80 backdrop-blur-xl p-8 rounded-2xl shadow-xl border border-white/50">

                {/* Pink Gradient Card Inner */}
                <div
                    className={`
            rounded-2xl border-2 border-dashed transition-all duration-300
            flex flex-col items-center justify-center p-10 text-center gap-6
            ${isDragging ? 'border-pink-500 bg-pink-50' : 'border-pink-200 bg-gradient-to-b from-pink-50 to-white'}
          `}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                >
                    <div className="w-16 h-16 bg-white rounded-full shadow-sm flex items-center justify-center text-3xl text-pink-500">
                        📄
                    </div>

                    <div>
                        <h3 className="text-xl font-bold text-gray-800 mb-1">
                            {file ? file.name : "Drop your .txt file here!"}
                        </h3>
                        <p className="text-sm text-gray-500">
                            {file ? "Ready to upload" : "Or click to browse and upload your personality data"}
                        </p>
                    </div>

                    <input
                        type="file"
                        accept=".txt"
                        onChange={handleFileChange}
                        className="hidden"
                        id="file-upload"
                    />

                    <label
                        htmlFor="file-upload"
                        className="btn-primary cursor-pointer flex items-center gap-2"
                    >
                        {uploading ? (
                            <>
                                <span className="animate-spin">⏳</span> Uploading...
                            </>
                        ) : (
                            <>
                                Choose File ✨
                            </>
                        )}
                    </label>

                    {file && !uploading && (
                        <button
                            onClick={handleUpload}
                            className="text-sm font-semibold text-pink-600 hover:text-pink-700 underline"
                        >
                            Start Analysis
                        </button>
                    )}
                </div>

                {error && (
                    <div className="mt-4 p-3 bg-red-50 text-red-600 text-sm rounded-lg text-center">
                        {error}
                    </div>
                )}
            </div>
        </div>
    );
};

export default FileUpload;
