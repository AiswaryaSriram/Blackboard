function construct_video_col(courseid,videotitle,videodesc)
{
	col=document.createElement("div");
	col.className="col-md-4";

	card_chart=document.createElement("div")
	card_chart.className="card card-chart";
	col.appendChild(card_chart);

	card_body=document.createElement("div");
	card_body.className="card-body";
	card_header=document.createElement("div");
	card_header.className="card-header card-header-"+card_type;
	card_chart.appendChild(card_header);
	card_chart.appendChild(card_body);

	h4=document.createElement("h4");
	h4.className="card-title";
	a=document.createElement("a");
	a.href="#";
	a.addEventListener("click",gotoCoursePage,false);
	a.innerHTML=name;
	h4.appendChild(a);

	card_header.appendChild(h4);

	p=document.createElement("p");
	p.className="category";
	p.innerHTML=teacher;
	card_header.appendChild(p);

	h4body=document.createElement("h4");
	h4body.className="card-title";
	h4body.innerHTML=desc;
	pbody=document.createElement("p");
	pbody.className="card-category";
	pbody.innerHTML="Course ID "+courseid;
	card_body.appendChild(pbody);
	card_body.appendChild(h4body);
	
	return col;
	

}
function videosSuccess()
{
	if(this.status==200 && this.readyState==4)
	{

		alert("loading videos");
		json=JSON.parse(this.response);
		videos=json['videos'];
		for(x in videos)
		{
			loc="videos/"+x;
			title=videos[x][videotitle];
			desc=videos[x][videodesc];

			alert(loc + "   "+title);
		}

	}
}
function gotoCoursePage(e) {
	e.preventDefault();
	courseid=e.target.parentNode.parentNode.nextSibling.firstChild.innerHTML.split('Course ID ')[1];
	alert(courseid);
	alert("00000still in gotoCorsePage method of check_status")
	window.location="course.html";
	xhr= new XMLHttpRequest();
	alert("still in gotoCorsePage method of check_status")
	xhr.open("POST", "http://localhost:5000/coursevideos", true);
	xhr.setRequestHeader("Content-type","application/x-www-form-urlencoded");
	xhr.onreadystatechange=videosSuccess;
	string_of_data="courseid="+courseid;
		//alert(string_of_data);
	xhr.send(string_of_data);


	// body...
}
function construct_course_col(courseid,name,desc,teacher,card_type)
{
	col=document.createElement("div");
	col.className="col-md-4";

	card_chart=document.createElement("div")
	card_chart.className="card card-chart";
	col.appendChild(card_chart);

	card_body=document.createElement("div");
	card_body.className="card-body";
	card_header=document.createElement("div");
	card_header.className="card-header card-header-"+card_type;
	card_chart.appendChild(card_header);
	card_chart.appendChild(card_body);

	h4=document.createElement("h4");
	h4.className="card-title";
	a=document.createElement("a");
	a.href="#";
	a.addEventListener("click",gotoCoursePage,false);
	a.innerHTML=name;
	h4.appendChild(a);

	card_header.appendChild(h4);

	p=document.createElement("p");
	p.className="category";
	p.innerHTML=teacher;
	card_header.appendChild(p);

	h4body=document.createElement("h4");
	h4body.className="card-title";
	h4body.innerHTML=desc;
	pbody=document.createElement("p");
	pbody.className="card-category";
	pbody.innerHTML="Course ID "+courseid;
	card_body.appendChild(pbody);
	card_body.appendChild(h4body);
	
	return col;
	

}
function success()
{
	if(this.status==200 && this.readyState==4)
	{
		//alert("fetched courses");
		//alert(this.response)
		json=JSON.parse(this.response)

		row=document.createElement("div");
		row.className="row";
		main=document.getElementById("main_content");
		main.appendChild(row);
		types=["success","warning","danger"]
		ct=0;
		offset=0;
		for (x in json)
		{

			course=json[x];
			card_type=types[ct+offset];
			col=construct_course_col(course['courseid'],course['course_name'],course['course_desc'],course['course_teacher'],card_type);
			row.appendChild(col);
			ct=(ct+1)%3;
		}
	}
}
function get_courses(email, encodedpwd)
{
	xhr= new XMLHttpRequest();
	xhr.open("POST", "http://localhost:5000/usercourses", true);
	xhr.setRequestHeader("Content-type","application/x-www-form-urlencoded");
	xhr.onreadystatechange=success;
	string_of_data="email="+email+"&password="+encodedpwd;
		//alert(string_of_data);
	xhr.send(string_of_data); 
}
function check_user_status()
{
	email=localStorage.getItem("email");
	password=localStorage.getItem("password");
	status=localStorage.getItem("status");
	alert("status"+status);
	if(status=="logged_in")
	{
		alert("user is logged in\n");
		get_courses(email,password);
		return true;
	}

}
check_user_status()
