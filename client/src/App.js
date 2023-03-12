import './App.css';
import { Fragment, useEffect } from 'react';
import {BrowserRouter as Router, Routes, Route,   } from 'react-router-dom';

import setAuthToken from './utils/setAuthToken';
//components
import PrivateRoutes from './components/routing/PrivateRoutes';
import PrivateCompanyRoutes from './components/routing/PrivateCompanyRoutes';
import EditProfile from './components/profile-forms/EditProfile';
import Profiles from './components/profiles/Profiles';
import Profile from './components/profile/Profile';
import Posts from './components/posts/Posts';
import AddExperience from './components/profile-forms/AddExperience';
import AddEducation from './components/profile-forms/AddEducation';
import Post from './components/posts/Post';
import PostJob from './components/job/PostJob';
import Login from './components/auth/Login';
import Register from './components/auth/Register';
import Dashboard from './components/dashboard/Dashboard';
import CompanyDashboard from './components/dashboard/CompanyDashboard';
import CreateProfile from './components/profile-forms/CreateProfile';
import Alert from './components/layout/Alert';
import Navbar from './components/layout/Navbar';
import Landing from './components/layout/Landing';
import Jobs from './components/job/Jobs';
import Job from './components/job/Job';
import EditJob from './components/job/EditJob';
import CreateCompanyProfile from './components/profile-forms/CreateCompanyProfile';
import EditCompanyProfile from './components/profile-forms/EditCompanyProfile';

// Redux
import { Provider } from 'react-redux'
import store from './store'
//actions
import { loadUser } from './actions/auth';
import { loadCompany } from './actions/auth';


if (localStorage.token){
  setAuthToken(localStorage.token)
}

function App() {
  useEffect(()=>{
    store.dispatch(loadUser())
    store.dispatch(loadCompany())
  },[])


  return (
    <Provider store={store}>
      <Router>
        <Fragment>
          <Navbar />
          <div className='alert-container'>
            <Alert />
          </div>
            <Routes>
              <Route exact path="/" element={<Landing />} />
              <Route exact path="/login" element={<Login />} />
              <Route exact path="/register" element={<Register />} />
              <Route exact path="/profiles" element={<Profiles />} />
              <Route exact path="/profile/:id" element={<Profile />} />
              <Route exact path="/jobs" element={<Jobs />} />
              <Route exact path="/job/:id" element={<Job />} />
              <Route element={<PrivateRoutes/>} >
                <Route path="/dashboard" element={<Dashboard />} />
                <Route path="/create-profile" element={<CreateProfile />} />
                <Route path="/edit-profile" element={<EditProfile />} />
                <Route path="/add-experience" element={<AddExperience />} />
                <Route path="/add-education" element={<AddEducation />} />
                <Route path="/posts" element={<Posts />} />
                <Route path="/posts/:id" element={<Post />} />
              </Route>
              <Route element={<PrivateCompanyRoutes />} >
                <Route path="/company-dashboard" element={<CompanyDashboard />} />
                <Route path="/post-job" element={<PostJob />} />
                <Route path="/edit-job/:id" element={<EditJob />} />
                <Route path="/create-company-profile" element={<CreateCompanyProfile />} />
                <Route path="/edit-company-profile" element={<EditCompanyProfile />} />
              </Route>
            </Routes>
        </Fragment>
      </Router>
    </Provider>
  );
}

export default App;
