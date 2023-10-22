const autoscroll = (id) => {
    console.log(id,"hello");
    var target = document.getElementsByClassName(id)[0];
    console.log(target);
    if(target){
        console.log("kslfj");
      const yOffset = target.getBoundingClientRect().top;

      window.scrollTo({
        top: window.scrollY + yOffset,
        behavior: 'smooth'
      })
    }
}

export default autoscroll;