import React , {Fragment} from 'react'
import PropTypes from 'prop-types'
import {connect} from 'react-redux'
import Moment from 'react-moment'
import { deleteEducation } from '../../actions/profile'

const Education = ({education, deleteEducation}) => {

    const degrees = education.map(edu => (
        <tr key={edu._id}>
            <td>{edu.school}</td>
            <td className="hide-sm"> {edu.degree} </td>
            <td className="hide-sm"> {edu.fieldofstudy} </td>
            <td className="hide-sm">
                <Moment format='DD/MM/YYYY'>{edu.from}</Moment> - { edu.to === null ? (
                    ' Now'
                    ) : (
                <Moment format='DD/MM/YYYY'>
                    {edu.to}
                </Moment>) }
            </td>
            <td>
                <button onClick={() => deleteEducation(edu._id)} className="btn btn-danger">Delete</button>
            </td>
        </tr>
    ))
  return (
    <Fragment>
        <h2 className="my-2">Educational Experiences</h2>
        <table className="table">
            <thead>
                <tr>
                    <th>Institute</th>
                    <th className="hide-sm">Degree</th>
                    <th className="hide-sm">Field Of Study</th>
                    <th className="hide-sm">Date</th>
                    <th ></th>
                </tr>
            </thead>
            <tbody>
                {degrees}
            </tbody>
        </table>
    </Fragment>
  )
}

Education.propTypes = {
    education: PropTypes.array.isRequired,
    deleteEducation: PropTypes.func.isRequired
}

export default connect(null, { deleteEducation })(Education)