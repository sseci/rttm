import Header from "./components/Header";
import Footer from "./components/Footer";
import MainPage from "./components/Mainpage";
import MainPicture from "./components/MainPicture";



function App() {
  return (
    //In this div we defined the background color
    <div className="container-fluid"
    style={{
        backgroundColor: 'grey',
      }}>
      <div headerName = "header">
        <Header />
      </div>
      <Footer />
      <MainPage />
      <MainPicture />
    </div>
  );
}

export default App;
