import React, { useEffect, useState } from 'react';
import { connect } from 'react-redux';
import PropTypes from 'prop-types';
import { Link } from 'react-router-dom';
import { addJob, getJobs } from '../../actions/job';

const PostJob = ({addJob}) => {
  
  const [resume,setResume]=useState(null);
  
  
    const handleChange = (e) => {
      setResume(e.target.files[0]);
  }

  const handleSubmit = async e => {
    if (!resume) {
      alert("Please upload your resume.");
      return;
  }
    e.preventDefault();
    addJob(resume);
  };
  useEffect(() => {getJobs()},[])

  return (
    <div className=''>
      <div className='container'>
        <div className='ml-10 mb-12 bg-light p-12 shadow-lg rounded-lg '>
          <div className='flex flex-col items-center justify-center'>
            <h2 className='text-6xl font-semibold text-primary mb-6 text-center'>Resume Parser</h2>
            <div className='ml-32'>
              <label htmlFor="fileInput">
                Choose File:
                <input  className='btn btn-light' type="file" id="fileInput" onChange={handleChange} />
              </label>
            </div>
            <button className='btn btn-dark mt-8 w-40' onClick={handleSubmit}>
              Upload
            </button>
          </div>
          </div>
          </div>
          </div>
  )
}

PostJob.propTypes = {
    addJob: PropTypes.func.isRequired
}

export default connect(null, {addJob})(PostJob)
