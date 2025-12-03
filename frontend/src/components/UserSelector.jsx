import React from 'react';

const UserSelector = ({ users, onSelectUser }) => {
    return (
        <div className="w-full max-w-4xl mx-auto">
            <h2 className="text-2xl font-bold text-gray-800 mb-6 text-center">
                Select a User to Analyze
            </h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
                {users.map((user, index) => (
                    <button
                        key={index}
                        onClick={() => onSelectUser(user)}
                        className="
              group relative overflow-hidden bg-white p-6 rounded-xl shadow-sm hover:shadow-md 
              border border-gray-100 hover:border-pink-200 transition-all duration-300
              flex items-center gap-4 text-left
            "
                    >
                        <div className="w-12 h-12 rounded-full bg-gradient-to-br from-pink-100 to-purple-100 text-purple-600 flex items-center justify-center text-xl font-bold group-hover:scale-110 transition-transform">
                            {user.charAt(0).toUpperCase()}
                        </div>
                        <div>
                            <div className="font-semibold text-gray-800 group-hover:text-pink-600 transition-colors">
                                {user}
                            </div>
                            <div className="text-xs text-gray-500">Click to view timeline</div>
                        </div>

                        {/* Hover Effect Background */}
                        <div className="absolute inset-0 bg-gradient-to-r from-pink-50 to-purple-50 opacity-0 group-hover:opacity-100 transition-opacity -z-10"></div>
                    </button>
                ))}
            </div>
        </div>
    );
};

export default UserSelector;
