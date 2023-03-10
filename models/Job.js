const mongoose = require('mongoose')
const Schema = mongoose.Schema

const JobSchema = Schema({
    company: {
        type: Schema.Types.ObjectId,
        ref: 'companies'
    },
    title: {
        type: String,
        required: true
    },
    location:{
        type: String
    },
    description: {
        type: String,
        required: true
    },
    type: {
        type: String
    },
    gender: {
        type: String
    },
    qualification: {
        type: String
    },
    requiredSkills: {
        type: [String],
        required: true
    },
    salaryFrom: {
        type: Number
    },
    salaryTo: {
        type: Number
    },
    positions: {
        type: Number
    },
    applicants: [{
        user: {
            type: Schema.Types.ObjectId,
            ref: 'users'
        },
        name: {
            type: String
        },
        avatar: {
            type: String
        },
        date:{
            type: Date,
            default: Date.now
        }
    }],
    date: {
        type: Date,
        default: Date.now
    }
})

module.exports = Job = mongoose.model('job', JobSchema)