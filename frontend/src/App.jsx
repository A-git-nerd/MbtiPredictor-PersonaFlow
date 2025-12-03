import React, { useState } from 'react';
import FileUpload from './components/FileUpload';
import UserSelector from './components/UserSelector';
import TimelinePanel from './components/TimelinePanel';
import axios from 'axios';

function App() {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [users, setUsers] = useState([]);
  const [selectedUser, setSelectedUser] = useState(null);
  const [timeline, setTimeline] = useState(null);
  const [loadingTimeline, setLoadingTimeline] = useState(false);

  const handleUploadSuccess = (data) => {
    setUploadedFile(data.filename);
    setUsers(data.users);
    setSelectedUser(null);
    setTimeline(null);
  };

  const handleSelectUser = async (user) => {
    setSelectedUser(user);
    setLoadingTimeline(true);
    try {
      const response = await axios.post('http://localhost:5000/api/predict', {
        filename: uploadedFile,
        selected_user: user,
      });
      setTimeline(response.data.timeline);
    } catch (error) {
      console.error("Error fetching timeline:", error);
      alert("Failed to generate timeline.");
    } finally {
      setLoadingTimeline(false);
    }
  };

  const handleCloseTimeline = () => {
    setTimeline(null);
    setSelectedUser(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-pink-50 to-purple-50 font-sans text-gray-900 overflow-hidden relative">
      {/* Background Decor */}
      <div className="absolute top-0 right-0 w-1/2 h-full bg-gradient-to-l from-white/40 to-transparent pointer-events-none"></div>

      <div className="max-w-7xl mx-auto px-6 py-8 relative z-10">
        {/* Header */}
        <header className="mb-16 flex justify-between items-center">
          <div className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-pink-500 to-purple-600">
            PersonaFlow
          </div>

        </header>

        <div className="flex flex-col lg:flex-row items-center justify-between gap-12">
          {/* Left Content */}
          <div className="lg:w-1/2 space-y-8">
            <h1 className="text-5xl md:text-6xl font-extrabold leading-tight text-gray-900">
              Discover Your <br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-pink-500 to-purple-600">
                Timeline Personality
              </span>
            </h1>
            <p className="text-lg text-gray-600 max-w-lg">
              Upload your text data and let our AI analyze how your personality evolves across your timeline.
            </p>

            {/* Feature Icons */}
            <div className="flex gap-6 pt-4">
              {[
                { label: 'Analysts', color: 'bg-purple-100 text-purple-600', icon: '🧠' },
                { label: 'Diplomats', color: 'bg-green-100 text-green-600', icon: '💡' },
                { label: 'Sentinels', color: 'bg-blue-100 text-blue-600', icon: '🛡️' },
                { label: 'Explorers', color: 'bg-yellow-100 text-yellow-600', icon: '⚡' },
              ].map((item, idx) => (
                <div key={idx} className="flex flex-col items-center gap-2 group cursor-default">
                  <div className={`w-14 h-14 rounded-full flex items-center justify-center text-2xl ${item.color} shadow-sm group-hover:scale-110 transition-transform`}>
                    {item.icon}
                  </div>
                  <span className="text-xs font-medium text-gray-500">{item.label}</span>
                </div>
              ))}
            </div>

            <div className="space-y-3 pt-4">
              {[
                'AI-powered timeline personality analysis',
                'We don’t store your chats',
                'Completely free to use'
              ].map((text, idx) => (
                <div key={idx} className="flex items-center gap-3 text-gray-600">
                  <div className="w-5 h-5 rounded-full bg-green-500 flex items-center justify-center text-white text-xs">✓</div>
                  {text}
                </div>
              ))}
            </div>
          </div>

          {/* Right Content - Upload Card */}
          <div className="lg:w-5/12 w-full">
            <FileUpload onUploadSuccess={handleUploadSuccess} />
          </div>
        </div>

        {/* User Selection Area (Appears after upload) */}
        {users.length > 0 && (
          <div className="mt-16 animate-fade-in">
            <UserSelector users={users} onSelectUser={handleSelectUser} />
          </div>
        )}

        {/* Loading Overlay */}
        {loadingTimeline && (
          <div className="fixed inset-0 bg-black/40 backdrop-blur-sm flex justify-center items-center z-50 animate-fade-in">
            <div className="bg-white p-8 rounded-2xl shadow-2xl flex flex-col items-center gap-4">
              <div className="w-12 h-12 border-4 border-purple-200 border-t-purple-600 rounded-full animate-spin"></div>
              <p className="text-lg font-semibold text-gray-700">Analyzing Personality...</p>
            </div>
          </div>
        )}

        {/* Timeline Panel */}
        {timeline && (
          <TimelinePanel
            timeline={timeline}
            selectedUser={selectedUser}
            onClose={handleCloseTimeline}
          />
        )}
      </div>
    </div>
  );
}

export default App;
