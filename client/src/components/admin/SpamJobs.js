import React , {Fragment} from 'react'
import {Link} from 'react-router-dom'
import PropTypes from 'prop-types'
import {connect} from 'react-redux'

const SpamJobs = ({jobs}) => {
  return (
    <div>
      <h2 className="mt-3 text-primary">Jobs Marked As Spam</h2>
      <table className="table mt-1">
        <thead>
          <tr>
              <th>Job Title</th>
              <th className="hide-sm">Job Type</th>
              <th className="hide-sm"></th>
              <th className="hide-sm"></th>
              <th className="hide-sm"></th>
          </tr>
        </thead>
        <tbody>
          {jobs ? (
            jobs.map((job) => (
              <tr key={job._id}>
                      <td className=""> {job.title} </td>
                      <td className="hide-sm"> {job.type} </td>
                      <td className="hide-sm">
                      <Link className='btn btn-light' to={`/job/${job.job}`} >
                          <p>View Job</p>
                      </Link>
                      </td>
                      <td className="hide-sm">
                          <button className="btn btn-primary">Remove from spam</button>
                      </td>
                      <td className="hide-sm">
                          <button className="btn btn-danger">Delete Job</button>
                      </td>
                  </tr>
            ))
          ) : (
            <p>No Company has been marked spam</p>
          ) }
        </tbody>
      </table>
        
    </div>
  )
}

SpamJobs.propTypes = {
  jobs: PropTypes.array.isRequired
}

export default connect(null, {})(SpamJobs)